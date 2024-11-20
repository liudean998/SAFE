import logging
import os

import math
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch_geometric.data import Batch
import torch.nn.functional as F
import torch.nn.init as init
from typing import Optional

from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
import lmdb, pickle
from layers.embedding_block import AtomEmbedding, EdgeEmbedding, AtomEmbeddingTags

from ocpmodels.common.utils import (
    compute_neighbors,
    conditional_grad,
    get_max_neighbors_mask,
    get_pbc_distances,
    radius_graph_pbc,
    scatter_det,
)
torch.autograd.set_detect_anomaly(True)

def _standardize(kernel):
    """
    Makes sure that N*Var(W) = 1 and E[W] = 0
    """
    eps = 1e-6

    if len(kernel.shape) == 3:
        axis = [0, 1]  # last dimension is output dimension
    else:
        axis = 1

    var, mean = torch.var_mean(kernel, dim=axis, unbiased=True, keepdim=True)
    kernel = (kernel - mean) / (var + eps) ** 0.5
    return kernel
def he_orthogonal_init(tensor: torch.Tensor) -> torch.Tensor:
    """
    Generate a weight matrix with variance according to He (Kaiming) initialization.
    Based on a random (semi-)orthogonal matrix neural networks
    are expected to learn better when features are decorrelated
    (stated by eg. "Reducing overfitting in deep networks by decorrelating representations",
    "Dropout: a simple way to prevent neural networks from overfitting",
    "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks")
    """
    tensor = torch.nn.init.orthogonal_(tensor)

    if len(tensor.shape) == 3:
        fan_in = tensor.shape[:-1].numel()
    else:
        fan_in = tensor.shape[1]

    with torch.no_grad():
        tensor.data = _standardize(tensor.data)
        tensor.data *= (1 / fan_in) ** 0.5

    return tensor
class Dense(torch.nn.Module):
    """
    Combines dense layer with scaling for silu activation.

    Arguments
    ---------
    in_features: int
        Input embedding size.
    out_features: int
        Output embedding size.
    bias: bool
        True if use bias.
    activation: str
        Name of the activation function to use.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

        if isinstance(activation, str):
            activation = activation.lower()
        if activation in ["silu", "swish"]:
            self._activation = torch.nn.SiLU()
        elif activation is None:
            self._activation = torch.nn.Identity()
        else:
            raise NotImplementedError(
                "Activation function not implemented for GemNet (yet)."
            )
    def reset_parameters(self, initializer=he_orthogonal_init) -> None:
        initializer(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0)

    def forward(self, x):
        x = self.linear(x)
        x = self._activation(x)
        return x

def get_graphs_and_indices(data, cutoff, max_neighbors, is_relax=False):
    """ "Generate embedding and interaction graphs and indices."""
    if is_relax:
        data.pos = data.pos_relaxed
    edge_index, cell_offsets, neighbors = radius_graph_pbc(data, cutoff, max_neighbors)
    out = get_pbc_distances(
                data.pos,
                edge_index,
                data.cell,
                cell_offsets,
                neighbors,
                return_offsets=True,
                return_distance_vec=True,
            )
    edge_index = out["edge_index"]
    edge_dist = out["distances"]
    cell_offset_distances = out["offsets"]
    distance_vec = out["distance_vec"]
    graph = {
        'edge_index': edge_index,
        'distances': edge_dist,
        'vector': distance_vec,
        'cell_offsets': cell_offsets,
        'cell_offsets_distances': cell_offset_distances,
        'neighbors': neighbors
    }
    return graph

def edge_index_get_vector(data, graph):
    out = get_pbc_distances(
        data.pos_relaxed,
        graph['edge_index'],
        data.cell,
        graph['cell_offsets'],
        graph['neighbors'],
        return_offsets=True,
        return_distance_vec=True
    )
    edge_index = out["edge_index"]
    edge_dist = out["distances"]
    cell_offset_distances = out["offsets"]
    distance_vec = out["distance_vec"]
    # 挑选出所有distance>6的边，进行矫正
    cells = torch.repeat_interleave(data.cell, graph['neighbors'], dim=0)
    dis6_index = torch.where(edge_dist > 7)[0]
    cells = cells[dis6_index]
    big_dis_edge = distance_vec[dis6_index]
    # max_indices = torch.argmax(torch.abs(big_dis_edge), dim=1)
    # max_signs = torch.sign(big_dis_edge[torch.arange(big_dis_edge.size(0)), max_indices])
    # for index, value in enumerate(list(big_dis_edge)):
    #     # 选择最小的边
    #     row = big_dis_edge[index]
    #     cell = cells[index]
    #     max_index = max_indices[index]
    #     max_sign = max_signs[index]
    #     row_new = row - max_sign*cell[:,max_index]
    #     row_sum = torch.sum(torch.abs(row), dim=0)
    #     row_new_sum = torch.sum(torch.abs(row_new), dim=0)
    #     row_norm = torch.norm(row_new, dim=0)
    #     if row_norm > 7:
    #         max_index = torch.argmax(torch.abs(row_new), dim=0)
    #         max_sign = torch.sign(row_new[max_index])
    #         row_new = row_new - max_sign*cell[:, max_index]
    #         row_new_sum = torch.sum(torch.abs(row_new), dim=0)
    #     if row_new_sum < row_sum:
    #         big_dis_edge[index] = row_new
    origin_vectors = graph['vector']
    for index, value in enumerate(list(big_dis_edge)):
        # 保持和init结构的向量符号相同
        row = big_dis_edge[index]
        cell = cells[index]
        origin_vector = origin_vectors[dis6_index[index]]
        signs = torch.sign(origin_vector)
        row_signs = torch.sign(row)
        signs_diff = (signs - row_signs)/2
        row_new = row.clone()
        for _i in range(2):
            row_new = row_new.clone() + cell.t()[_i]*signs_diff[_i]
        row_new_norm = torch.norm(row_new, dim=0)
        if row_new_norm > 8:
            max_index = torch.argmax(torch.abs(row_new[:2]), dim=0)
            max_sign = torch.sign(row_new[max_index])
            row_new = row_new - max_sign*cell.t()[max_index]
            row_new_norm = torch.norm(row_new, dim=0)
            if row_new_norm > 8:
                max_index = torch.argmax(torch.abs(row_new[:2]), dim=0)
                max_sign = torch.sign(row_new[max_index])
                row_new = row_new - max_sign * cell.t()[max_index]
        # row_new_norm = torch.norm(row_new)
        # if row_new_norm > 9:
        #     print(cell, 'cell')
        #     print(row, 'row')
        #     print(row_new, 'row_new')
        #     print(origin_vector, 'origin')
        big_dis_edge[index] = row_new
    for _index, _value in enumerate(list(dis6_index)):
        distance_vec[_value] = big_dis_edge[_index]
        edge_dist[_value] = torch.norm(big_dis_edge[_index])
    graph = {
        'edge_index': edge_index,
        'distances': edge_dist,
        'vector': distance_vec,
        'cell_offsets': graph['cell_offsets'],
        'cell_offsets_distances': cell_offset_distances,
    }
    return graph

class InterBlock(torch.nn.Module):
    def __init__(self,
                 emb_size_in:int,
                 emb_size_out:int,
                 ):
        super().__init__()
        self.linear_a = nn.Linear(emb_size_in, emb_size_in)
        self.linear_b = nn.Linear(emb_size_in, emb_size_in)
        self.linear_c = nn.Linear(emb_size_in, emb_size_in)
        self.dense = torch.nn.Linear(emb_size_in, emb_size_out, bias=True)
        self._activation = torch.nn.LeakyReLU(negative_slope=0.01)
        torch.nn.init.xavier_uniform_(self.linear_a.weight, gain=0.1)
        torch.nn.init.xavier_uniform_(self.linear_b.weight, gain=0.1)
        torch.nn.init.xavier_uniform_(self.linear_c.weight, gain=0.1)

    def forward(self, v, edge_index, edges_per_node):
         # 和本向量相关的向量相乘 Z = a*X + sum(b_i*Y_i)
        for _index in range(len(edge_index)):
            # v_index = self.linear_a(v[_index].clone())
            v_index = v[_index].clone()
            # v1 = v_origin[_index]
            # v1_norm = torch.norm(v1)
            start_index = edge_index[0][_index]
            target_index = edge_index[1][_index]
            for _edge in edges_per_node[start_index]:
                if _edge != _index:
                    v_index = (v_index.clone() +
                               0.1*self.linear_a(v_index.clone()) +
                               0.1*self.linear_b(v[_edge].clone()))
            for _edge in edges_per_node[target_index]:
                if _edge != _index:
                    v_index = (v_index.clone() +
                               0.1 * self.linear_a(v_index.clone()) +
                               0.1 * self.linear_b(v[_edge].clone()))
            v[_index] = v_index
        update_v = self.dense(v)
        update_v = self._activation(update_v)
        return update_v

class ResidualLayer(torch.nn.Module):
    """
    Residual block with output scaled by 1/sqrt(2).

    Arguments
    ---------
    units: int
        Input and output embedding size.
    nLayers: int
        Number of dense layers.
    layer: torch.nn.Module
        Class for the layers inside the residual block.
    layer_kwargs: str
        Keyword arguments for initializing the layers.
    """

    def __init__(
        self, units: int, nLayers: int = 2, layer=Dense, **layer_kwargs
    ) -> None:
        super().__init__()
        self.dense_mlp = torch.nn.Sequential(
            *[
                layer(
                    in_features=units,
                    out_features=units,
                    bias=False,
                    **layer_kwargs
                )
                for _ in range(nLayers)
            ]
        )
        self.inv_sqrt_2 = 1 / math.sqrt(2)

    def forward(self, input):
        x = self.dense_mlp(input)
        x = input + x
        x = x * self.inv_sqrt_2
        return x

class OutBlock(nn.Module):
    def __init__(self, edge_radial_in, edge_radial_out,
        nHidden: int,
        activation: Optional[str] = None,) -> None:
        super(OutBlock, self).__init__()
        self.layers = self.get_mlp(edge_radial_in, edge_radial_out, nHidden, activation)
        self.rbf = nn.Linear(edge_radial_in, edge_radial_out, bias=False)
        self.scale_rbf = nn.Linear(edge_radial_in, edge_radial_out, bias=False)

    def get_mlp(self, units_in: int, units: int, nHidden: int, activation):
        if units_in != units:
            dense1 = Dense(units_in, units, activation=activation, bias=False)
            mlp = [dense1]
        else:
            mlp = []
        res = [
            ResidualLayer(units, nLayers=2, activation=activation)
            for _ in range(nHidden)
        ]
        mlp += res
        return torch.nn.ModuleList(mlp)

    def forward(self, m, v_rad):
        v_change = m.clone()  # 使用 clone() 创建副本，避免原地修改
        for layer in self.layers:
            v_change = layer(v_change)  # 传递修改后的副本
        basis_emb_V = self.rbf(v_rad)
        V_basis = v_change * basis_emb_V
        V = self.scale_rbf(V_basis)
        return V


class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATLayer, self).__init__()
        self.attn = nn.MultiheadAttention(in_channels, num_heads=1)
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, v):
        # 获取边的特征向量 (v 是边的特征，形状为 [N, 512])
        v_edge = v  # 直接使用边的特征

        # 这里 v_edge 需要转换成 (seq_len, batch_size, embedding_dim) 的形状
        # 因为 MultiheadAttention 需要的是 [N, 1, 512] 格式的输入
        v_edge = v_edge.unsqueeze(1)  # 变成 (N, 1, 512)

        # 使用 MultiheadAttention 进行边特征的聚合
        attn_output, _ = self.attn(v_edge, v_edge, v_edge)  # 注意力机制

        # 通过全连接层变换输出的特征
        output = self.fc(attn_output.squeeze(1))  # 需要去掉 batch_size 维度，变为 (N, out_channels)

        return output

class GaussianBasis(torch.nn.Module):
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        if trainable:
            self.offset = torch.nn.Parameter(offset, requires_grad=True)
        else:
            self.register_buffer("offset", offset)
        self.coeff = -0.5 / ((stop - start) / (num_gaussians - 1)) ** 2

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist[:, None] - self.offset[None, :]
        return torch.exp(self.coeff * torch.pow(dist, 2))

class ExponentialEnvelope(torch.nn.Module):
    """
    Exponential envelope function that ensures a smooth cutoff,
    as proposed in Unke, Chmiela, Gastegger, Schütt, Sauceda, Müller 2021.
    SpookyNet: Learning Force Fields with Electronic Degrees of Freedom
    and Nonlocal Effects
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, d_scaled: torch.Tensor) -> torch.Tensor:
        env_val = torch.sign(d_scaled)*torch.exp(
            -(d_scaled**2) / ((1 - d_scaled) * (1 + d_scaled))
        )
        return torch.where(d_scaled < 2, env_val, torch.zeros_like(d_scaled))

class RadialBasis(torch.nn.Module):
    """

    Arguments
    ---------
    num_radial: int
        Number of basis functions. Controls the maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    rbf: dict = {"name": "gaussian"}
        Basis function and its hyperparameters.
    envelope: dict = {"name": "polynomial", "exponent": 5}
        Envelope function and its hyperparameters.
    scale_basis: bool
        Whether to scale the basis values for better numerical stability.
    """

    def __init__(
        self,
        num_radial: int,
        cutoff: float,
        scale_basis: bool = False,
    ) -> None:
        super().__init__()
        self.inv_cutoff = 1 / cutoff

        self.scale_basis = scale_basis

        self.envelope = ExponentialEnvelope()
        rbf = {"name": "gaussian"}
        rbf_name = rbf["name"].lower()
        rbf_hparams = rbf.copy()
        del rbf_hparams["name"]

        # RBFs get distances scaled to be in [0, 1]
        self.rbf = GaussianBasis(
                start=0, stop=1, num_gaussians=num_radial, **rbf_hparams
            )

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        d_scaled = d * self.inv_cutoff
        env = self.envelope(d_scaled)
        res = env[:, None] * self.rbf(d_scaled)
        return res

class LMDBDataset(Dataset):
    def __init__(self, lmdb_path):
        self.lmdb_path = lmdb_path
        # 打开 LMDB 文件
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            # 获取数据集中总样本数
            self.length = txn.stat()['entries']

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            # 读取数据
            byte_data = txn.get(f"{index}".encode())
            data = pickle.loads(byte_data)  # 假设数据已用 torch 序列化保存
        return data


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, x, y):
        row_sums = torch.sum(torch.abs(y-x), dim=1)
        avg = torch.mean(row_sums)
        return avg

class MatrixTransformer(nn.Module):
    def __init__(self,
                 input_dim=3,
                 cutoff=6,
                 edge_radial=512,
                 atom_radial=256,
                 num_blocks=3,
                 activation="silu",
                 num_global_out_layers=2):
        super(MatrixTransformer, self).__init__()
        self.edges_per_node = None
        # 原子和边嵌入
        self.atom_emb = AtomEmbedding(atom_radial,83)
        self.rbf = RadialBasis(edge_radial, cutoff)
        self.rbf_scaler = nn.Linear(edge_radial*3, edge_radial)
        self.vector_fc1 = nn.Linear(input_dim, edge_radial)
        self.vector_activation = nn.ReLU()
        self.bond_emb = EdgeEmbedding(atom_radial, edge_radial, edge_radial)
        self.vector_fc2 = nn.Linear(input_dim, edge_radial)
        self.num_blocks = num_blocks
        int_blocks = []
        for _ in range(num_blocks):
            int_blocks.append(InterBlock(edge_radial, edge_radial))
        self.inter_block = torch.nn.ModuleList(int_blocks)
        out_blocks = []
        for _ in range(num_blocks+1):
            out_blocks.append(OutBlock(edge_radial, edge_radial, nHidden=3, activation=activation))
        self.out_block = torch.nn.ModuleList(out_blocks)
        out_mlp_F = [
                        Dense(
                            (edge_radial) * (num_blocks + 1),
                            edge_radial,
                            activation=activation,
                        )
                    ] + [
                        ResidualLayer(
                            edge_radial,
                            activation=activation,
                        )
                        for _ in range(num_global_out_layers)
                    ]
        self.out_mlp_F = torch.nn.Sequential(*out_mlp_F)
        self.out_v = Dense(edge_radial, 3, bias=False, activation=None)
        self.out_act = torch.nn.Tanh()
        # self.inter_block = GATLayer(edge_radial*7, edge_radial*7)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用 Xavier 初始化
                init.xavier_uniform_(m.weight)

    def get_edges_per_node(self, edge_index, natoms):
        # 统计每个节点相关的边
        edges_per_node = [[] for _ in range(natoms)]
        num_edges = edge_index.size(1)
        for i in range(num_edges):
            source = edge_index[0, i].item()  # 源节点
            target = edge_index[1, i].item()  # 目标节点
            # 将边的索引添加到源节点和目标节点的列表中
            edges_per_node[source].append(i)  # 存储边的索引
            edges_per_node[target].append(i)  # 存储边的索引
        self.edges_per_node = edges_per_node

    def forward(self, graph, batch):
        v = graph['vector']
        edge_index = graph['edge_index']
        dist = graph['distances']
        natoms = len(batch.atomic_numbers.tolist())
        self.get_edges_per_node(edge_index, natoms)
        new_v = v.clone()
        new_v.requires_grad_(True)
        # v shape 是 (N, 3)，直接传入每行3维向量
        nummm = 2
        for i in range(nummm):
            # print(torch.isnan(new_v).any(),torch.isinf(new_v.any()), i)
            h = self.atom_emb(batch.atomic_numbers.long())
            dis_rbf = self.rbf(dist)
            # print(torch.isnan(m).any(), torch.isinf(m.any()), 'rbf',i)
            # m1 = self.rbf_scaler(m1.clone())
            # m1 = self.vector_fc1(new_v.clone())
            # print(m2.shape)
            # m1 = self.vector_activation(m1.clone())
            m = self.bond_emb(h, dis_rbf, edge_index)

            m_rad = self.vector_fc2(new_v)
            delta = []
            delta0 = self.out_block[0](m.clone(), m_rad)
            # print(torch.isnan(delta0.any()), 'delta0')
            delta.append(delta0)
            for _i in range(self.num_blocks):
                m_inter = self.inter_block[i](m.clone(), edge_index, self.edges_per_node)
                # print(f"Iteration {i}, m contains NaN: {torch.isnan(m).any()}")
                delta_i = self.out_block[i+1](m_inter.clone(), m_rad)
                # print(f"Iteration {i}, delta_{_i} contains NaN: {torch.isnan(delta_i).any()}")
                delta.append(delta_i)
            # print(f"Iteration {i}, delta contains NaN: {torch.isnan(torch.cat(delta, dim=1)).any()}")
            delta_X = self.out_mlp_F(torch.cat(delta, dim=1))
            # print(f"Iteration {i}, delta_X contains NaN: {torch.isnan(delta_X).any()}")
            delta_X_out = self.out_v(delta_X)
            delta_X_out = self.out_act(delta_X_out.clone())*6
            # print(torch.max(delta_X_out), torch.min(delta_X_out), torch.mean(delta_X_out))
            new_v = new_v + delta_X_out.clone()/(2-i)
            # print(f"Iteration {i}, new_v contains NaN: {torch.isnan(delta_i).any()}")
        return new_v  # 输出和输入形状一致 (N, 3)

def main(lmdb_path, batch_size, epoch, val_path):
    train_dataset = LMDBDataset(lmdb_path)
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataset = LMDBDataset(val_path)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    # Initialize model, optimizer, and loss function
    model = MatrixTransformer()
    base_lr = 0.001
    other_params = [
        param for name, param in model.named_parameters()
        if "fc_out" not in name and "inter_block" not in name
    ]
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    criterion = nn.L1Loss()
    # criterion = nn.SmoothL1Loss()
    # criterion = CustomLoss()

    # 统计参数总数
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'总参数量: {total_params}')
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    loss_best = float('inf')

    check_path = f'./result/{formatted_time}/checkpoint.pth'
    os.makedirs(os.path.dirname(check_path), exist_ok=True)

    # Training loop
    for e in range(epoch):
        num = len(data_loader)
        for iii, batch in enumerate(data_loader):
            print(((e * num)+(iii + 1)))
            optimizer.zero_grad()
            model.train()
            X_graph = get_graphs_and_indices(batch, cutoff=6, max_neighbors=8)
            X = X_graph['vector']
            Y_graph = edge_index_get_vector(batch, X_graph)
            Y = Y_graph['vector']
            # continue
            X_transformed = model(X_graph, batch)
            loss = criterion(Y, X_transformed)

            # Backpropagation

            loss.backward()
            # # 使用梯度裁剪，限制最大梯度范数为 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # print((epoch + 1) % 100)
            if (iii + 1) % 10 == 0:
                print(f'Train:::Epoch [{((e * num)+(iii + 1)) *100/(num*epoch):.4f}%], Loss: {loss.item():.4f}')
                print(criterion(X,Y))
            if (iii + 1) % 100 == 0:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"{name} gradient mean: {param.grad.mean().item()}")
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        print(f"{name} is frozen")
            if ((e * num)+(iii + 1)) % 1000 == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    count = 0
                    for val_b in val_loader:
                        X_graph = get_graphs_and_indices(val_b, cutoff=6, max_neighbors=8)
                        # print(X_graph)
                        X = X_graph['vector']
                        X_transformed = model(X, X_graph['edge_index'], val_b)
                        Y_graph = edge_index_get_vector(val_b, X_graph)
                        Y = Y_graph['vector']
                        loss = criterion(X_transformed, Y)
                        # print(val_loss)
                        val_loss += loss.item()*Y.size()[0]
                        count += Y.size()[0]
                    val_loss /= count
                    if val_loss < loss_best:
                        loss_best = val_loss
                        torch.save(model.state_dict(), check_path)
                    print(f'Val:::Epoch [{(iii + 1)*(e + 1) * 1000 / (num * epoch)}%], Loss: {val_loss:.4f}')
                model.train()



if __name__ == '__main__':
    lmdb_p = '/media/liud/Liud_FX2T/dataset/OC20/is2re_10k/select_data/tag1&2/train'
    val_p = '/media/liud/Liud_FX2T/dataset/OC20/is2re_10k/select_data/tag1&2/test'
    main(lmdb_p, batch_size=6, epoch= 100, val_path = val_p)