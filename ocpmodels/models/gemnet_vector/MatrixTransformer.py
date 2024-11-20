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

def get_graphs_and_indices(data, cutoff, max_neighbors):
    """ "Generate embedding and interaction graphs and indices."""
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
                    # v_index = v_index.clone() + self.linear_b(v[_edge].clone())
                    # v2 = v_origin[_edge]
                    # dot_product = torch.matmul(v1, v2)
                    # v2_norm = torch.norm(v2)
                    # cos_theta = dot_product / (v2_norm*v1_norm)
                    # if cos_theta <= -1:
                    #     cos_theta = torch.tensor(-1.0)
                    # elif cos_theta >= 1:
                    #     cos_theta = torch.tensor(1.0)
                    # theta = torch.acos(cos_theta).item()
                    # angle_aaa = 1-(theta/3.2)
                    # v_index = self.linear_a(v_index.clone()) + self.linear_b(angle_aaa*v[_edge].clone())
                    v_index = (v_index.clone() +
                               0.1*self.linear_a(v_index.clone()) +
                               0.1*self.linear_b(v[_edge].clone()))
            for _edge in edges_per_node[target_index]:
                if _edge != _index:
                    # v2 = v_origin[_edge]
                    # dot_product = torch.matmul(v1, v2)
                    # v2_norm = torch.norm(v2)
                    # cos_theta = dot_product / (v2_norm * v1_norm)
                    # if cos_theta <= -1:
                    #     cos_theta = torch.tensor(-1.0)
                    # elif cos_theta >= 1:
                    #     cos_theta = torch.tensor(1.0)
                    # theta = torch.acos(cos_theta).item()
                    # angle_aaa = 1 - (theta / 3.2)
                    # # v_index = v_index.clone() + self.linear_b(v[_edge].clone())
                    # v_index = self.linear_a(v_index.clone()) + self.linear_b(angle_aaa*v[_edge].clone())
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
        init.xavier_normal_(self.rbf.weight)
        init.xavier_normal_(self.scale_rbf.weight)

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
        dist = dist[:, :, None] - self.offset[None, None, :]
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
        return torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))

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
        res = env[:,:, None] * self.rbf(d_scaled)
        return res

class MatrixTransformer(nn.Module):
    def __init__(self, input_dim=3, cutoff=6, edge_radial=64, atom_radial=128,
                 num_blocks=3,
                 activation="silu",
                 num_global_out_layers=2):
        super(MatrixTransformer, self).__init__()
        self.edges_per_node = None
        # 原子和边嵌入
        self.atom_emb = AtomEmbedding(atom_radial,83)
        self.rbf = RadialBasis(edge_radial, cutoff)
        self.vector_fc1 = nn.Linear(input_dim, edge_radial*3)
        # self.vector_activation = nn.ReLU()
        self.bond_emb = EdgeEmbedding(128, edge_radial*3, edge_radial*7)

        self.vector_fc2 = nn.Linear(edge_radial*3, edge_radial*7)
        # 初始化参数
        torch.nn.init.xavier_uniform_(self.vector_fc1.weight)
        torch.nn.init.kaiming_uniform_(self.vector_fc2.weight)

        # 线性层的输入维度调整为3，输出维度可以根据需要设置
        # self.dropout2 = nn.Dropout(p=0.2)
        # self.dropout4 = nn.Dropout(p=0.4)
        # self.fc1 = nn.Linear(edge_radial*7, 512)
        # # self.bn1 = nn.BatchNorm1d(1024)
        # self.swish = nn.Hardswish()
        # self.fc2 = nn.Linear(512, 512)
        # # self.bn2 = nn.BatchNorm1d(512)
        # self.tanh = nn.Tanh()
        # self.gelu2 = nn.GELU()
        # self.fc3 = nn.Linear(512, 256)
        # self.gelu3 = nn.GELU()
        # self.fc_out = nn.Linear(256, input_dim)  # 输出维度为3，和输入保持一致
        self.num_blocks = num_blocks
        int_blocks = []
        for _ in range(num_blocks):
            int_blocks.append(InterBlock(edge_radial*7, edge_radial*7))
        self.inter_block = torch.nn.ModuleList(int_blocks)
        out_blocks = []
        for _ in range(num_blocks+1):
            out_blocks.append(OutBlock(edge_radial*7, edge_radial*7, nHidden=3, activation=activation))
        self.out_block = torch.nn.ModuleList(out_blocks)
        out_mlp_F = [
                        Dense(
                            (edge_radial*7) * (num_blocks + 1),
                            edge_radial*7,
                            activation=activation,
                        )
                    ] + [
                        ResidualLayer(
                            edge_radial*7,
                            activation=activation,
                        )
                        for _ in range(num_global_out_layers)
                    ]
        self.out_mlp_F = torch.nn.Sequential(*out_mlp_F)
        self.out_v = Dense(edge_radial*7, 3, bias=False, activation=None)
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



    def forward(self, v, edge_index, batch):
        natoms = len(batch.atomic_numbers.tolist())
        self.get_edges_per_node(edge_index, natoms)
        new_v = v.clone()
        # v shape 是 (N, 3)，直接传入每行3维向量
        nummm = 2
        for i in range(nummm):
            # print(torch.isnan(new_v).any(),torch.isinf(new_v.any()), i)
            h = self.atom_emb(batch.atomic_numbers.long())
            # m = self.rbf(new_v.clone())
            # print(torch.isnan(m).any(), torch.isinf(m.any()), 'rbf',i)
            # m = m.view(m.size(0), -1)
            m1 = self.vector_fc1(new_v.clone())
            # m = self.vector_activation(m.clone())
            m = self.bond_emb(h, m1, edge_index)

            m_rad = self.vector_fc2(m1)
            delta = []
            delta0 = self.out_block[0](m, m_rad)
            delta.append(delta0)
            for _i in range(self.num_blocks):
                m = self.inter_block[i](m, edge_index, self.edges_per_node)
                delta_i = self.out_block[i+1](m, m_rad)
                # print(f"Iteration {i}, delta_{_i} contains NaN: {torch.isnan(delta_i).any()}")
                delta.append(delta_i)
            # print(f"Iteration {i}, delta contains NaN: {torch.isnan(torch.cat(delta, dim=1)).any()}")
            delta_X = self.out_mlp_F(torch.cat(delta, dim=1))
            # print(f"Iteration {i}, delta_X contains NaN: {torch.isnan(delta_X).any()}")
            delta_X_out = self.out_v(delta_X)
            new_v = new_v + delta_X_out.clone()
            # After updating new_v
            # print(f"Iteration {i}, new_v contains NaN: {torch.isnan(new_v).any()}")
            # out1 = self.fc1(m)
            # # out2 = self.bn1(out1)
            # out3 = self.swish(out1)
            # out4 = self.dropout4(out3)
            #
            # # out5 = self.fc2(out4)
            # # out6 = self.bn2(out5)
            # # out8 = self.gelu2(out5)
            # # out6 = self.dropout4(out5)
            # out5 = self.fc3(out4)
            # out9 = self.gelu3(out5)
            # out10 = self.gelu3(out9)
            # new_v = self.fc_out(out10)
            # # delta_X = new_X-v
            # # # new_v = new_v + delta_X*((i+1)/3)
            # # # delta_X = self.fc_out(out10)
            # # if delta_X_all is None:
            # #     delta_X_all = torch.zeros_like(delta_X)
            # # delta_X_all += delta_X
            # # 并行三层
            # # delta_X_all = None
            # # for lll in range(2):
            # #     out1 = self.fc1(m)
            # #     # out2 = self.bn1(out1)
            # #     out3 = self.swish(out1)
            # #     out4 = self.dropout4(out3)
            # #
            # #     # out5 = self.fc2(out4)
            # #     # out6 = self.bn2(out5)
            # #     # out8 = self.gelu2(out5)
            # #     # out6 = self.dropout4(out5)
            # #     out5 = self.fc3(out4)
            # #     out9 = self.gelu3(out5)
            # #     out10 = self.gelu3(out9)
            # #     new_X = self.fc_out(out10)
            # #     delta_X = new_X-v
            # #     # new_v = new_v + delta_X*((i+1)/3)
            # #     # delta_X = self.fc_out(out10)
            # #     if delta_X_all is None:
            # #         delta_X_all = torch.zeros_like(delta_X)
            # #     delta_X_all += delta_X
            # # new_v = new_v + delta_X_all*((i+1)/3*nummm)
        return new_v  # 输出和输入形状一致 (N, 3)

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
def main(lmdb_path, batch_size, epoch, val_path):
    train_dataset = LMDBDataset(lmdb_path)
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    val_dataset = LMDBDataset(val_path)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    # Initialize model, optimizer, and loss function
    model = MatrixTransformer()
    base_lr = 0.0001
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
            # print(X_graph)
            X = X_graph['vector']
            # print(X)
            # Predict delta_X
            # X_transformed = model(X, X_graph['edge_index'], batch)
            # Updated matrix after applying the predicted transformation
            # print(X_transformed)

            Y_graph = edge_index_get_vector(batch, X_graph)
            Y = Y_graph['vector']
            print(X)
            print(Y)
            import time
            time.sleep(10000)
            # delta = Y - X
            # # 先统计一下两个结构向量差值的平均值是多少
            # row_sums = torch.sum(torch.abs(delta), dim=1)
            # avg = torch.mean(row_sums)
            # print(X_graph, Y_graph)
            # import time
            # time.sleep(10000)
            # Compute loss
            loss = criterion(Y, X_transformed)

            # Backpropagation

            loss.backward()
            # # 使用梯度裁剪，限制最大梯度范数为 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # print((epoch + 1) % 100)
            if (iii + 1) % 10 == 0:
                print(f'Train:::Epoch [{((e * num)+(iii + 1)) *100/(num*epoch):.4f}%], Loss: {loss.item():.4f}')
                # print(X, Y, X_transformed, delta)
                print(criterion(X,Y))
                # print(avg)
                # row_sums1 = torch.sum(torch.abs(Y - X_transformed), dim=1)
                # avg1 = torch.mean(row_sums1)
                # print(avg1)
                # import time
                # time.sleep(10000)
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


from ase import Atoms
from ase.geometry import cell_to_cellpar, cellpar_to_cell
import numpy as np


def apply_minimum_image_convention(coord1, coord2, cell):
    """
    在周期性晶胞 cell 下找到 coord1 和 coord2 之间的最小镜像距离。

    Args:
    - coord1: numpy 数组，表示结构 A 的原子坐标
    - coord2: numpy 数组，表示结构 B 的原子坐标
    - cell: 晶胞矩阵，shape = (3, 3)

    Returns:
    - 最小镜像距离向量
    """
    atoms = Atoms(cell=cell,
                  numbers=[1,1],
                  positions=[coord1, coord2],
                  pbc=True)
    aaa = atoms.get_all_distances(mic=True, vector=True)
    return aaa[0][1]


# 假设两个结构为 struct1 和 struct2，均为 Atoms 对象
def minimize_structure_difference(lmdb_path):
    env = lmdb.open(lmdb_path, map_size=1099511627776, readonly=True, lock=False)
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            data = pickle.loads(value)
            atomic_numbers = data.atomic_numbers
            cell = data.cell[0]
            pos = data.pos
            pos_relaxed = data.pos_relaxed
            struct1 = Atoms(cell=cell,
                          numbers=atomic_numbers,
                          positions=pos,
                            pbc=True)
            struct2 = Atoms(cell=cell,
                            numbers=atomic_numbers,
                            positions=pos_relaxed,
                            pbc=True)
            diff1 = struct1.get_positions() - struct2.get_positions()
            diff2 = []
            for i in range(len(list(atomic_numbers))):
                diff_i = apply_minimum_image_convention(pos[i], pos_relaxed[i], cell)
                diff2.append(diff_i)
                if (np.linalg.norm(diff1[i]) - np.linalg.norm(diff_i))>0.5:
                    print(diff1[i], diff_i)
            # break


import torch
import lmdb
import pickle
import numpy as np
from ase import Atoms


def apply_minimum_image_convention_torch(pos, pos_relaxed, cell):
    """
    使用 PyTorch 计算周期性条件下的最小镜像向量。

    Args:
    - pos: 初始结构的原子位置，形状 (N, 3)
    - pos_relaxed: 经过弛豫后的原子位置，形状 (N, 3)
    - cell: 晶胞矩阵，形状 (3, 3)

    Returns:
    - 最小镜像向量的差异矩阵，形状 (N, 3)
    """
    # 将输入数据转换为 PyTorch 张量
    pos = torch.tensor(pos, dtype=torch.float32)
    pos_relaxed = torch.tensor(pos_relaxed, dtype=torch.float32)
    cell = torch.tensor(cell, dtype=torch.float32)

    # 计算原始坐标差异
    delta = pos - pos_relaxed

    # 将 delta 转换到分数坐标系
    frac_delta = torch.matmul(delta, torch.inverse(cell).T)

    # 应用最小镜像约定：将坐标调整到 [-0.5, 0.5) 的范围内
    frac_delta -= frac_delta.round()

    # 转换回晶胞坐标
    min_image_delta = torch.matmul(frac_delta, cell)

    return min_image_delta


def minimize_structure_difference1(lmdb_path):
    env = lmdb.open(lmdb_path, map_size=1099511627776, readonly=True, lock=False)
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            data = pickle.loads(value)
            atomic_numbers = data.atomic_numbers
            cell = data.cell[0]
            pos = data.pos
            pos_relaxed = data.pos_relaxed

            # 使用 PyTorch 计算周期性条件下的最小镜像向量差异
            diff_min_image = apply_minimum_image_convention_torch(pos, pos_relaxed, cell)

            # 计算每个原子的差异大小，并与原始差异进行比较
            diff_original = torch.tensor(pos) - torch.tensor(pos_relaxed)
            diff_norm_original = torch.norm(diff_original, dim=1)
            diff_norm_min_image = torch.norm(diff_min_image, dim=1)

            # 查找差异显著的原子对并输出
            significant_diff_indices = \
            (torch.abs(diff_norm_original - diff_norm_min_image) > 0.5).nonzero(as_tuple=True)[0]
            for i in significant_diff_indices:
                print("原始差异向量:", diff_original[i].numpy())
                print("最小镜像差异向量:", diff_min_image[i].numpy())


if __name__ == '__main__':
    lmdb_p = '/media/liud/Liud_FX2T/dataset/OC20/is2re_10k/select_data/all_stru/train'
    val_p = '/media/liud/Liud_FX2T/dataset/OC20/is2re_10k/select_data/tag1&2/test'
    main(lmdb_p, batch_size=4, epoch= 100, val_path = val_p)
    # minimize_structure_difference1(val_p)
