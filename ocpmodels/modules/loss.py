import logging
from typing import Optional

import torch
from torch import nn

from ocpmodels.common import distutils


class L2MAELoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        dists = torch.norm(input - target, p=2, dim=-1)
        if self.reduction == "mean":
            return torch.mean(dists)
        elif self.reduction == "sum":
            return torch.sum(dists)

class MinDist(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def minimum_image_convention_torch(self, pos, pos_relaxed, cell, ptr):
        """
        使用 PyTorch 计算周期性条件下的最小镜像向量。

        Args:
        - pos: 初始结构的原子位置，形状 (N, 3)
        - pos_relaxed: 经过弛豫后的原子位置，形状 (N, 3)
        - cell: 晶胞矩阵，形状 (3, 3)

        Returns:
        - 最小镜像向量的差异矩阵，形状 (N, 3)
        """
        from ase import Atoms
        min_dist = []
        for i in range(len(list(ptr))-1):
            pos_i = pos[ptr[i] : ptr[i+1]]
            pos_relaxed_i = pos_relaxed[ptr[i] : ptr[i+1]]
            cell_i = cell[i].cpu()
            pos_all= []
            for index, value in enumerate(list(pos_i)):
                pos_all.append(value.tolist())
                pos_all.append(pos_relaxed_i[index].tolist())
            atoms = Atoms(cell=cell_i,
                          numbers=[1]*len(pos_all),
                          positions=pos_all,
                          pbc=True)
            dist_vector = atoms.get_all_distances(mic=True, vector=True)
            dist_vector = torch.tensor(dist_vector, dtype=torch.float32, device=pos_i.device, requires_grad=True)
            for double_index in range(len(dist_vector)):
                if double_index % 2 == 0:
                    min_dist.append(dist_vector[double_index, double_index+1])
        min_dist = torch.stack(min_dist, dim=0)
        min_dist.requires_grad_(True)
        return min_dist

    def forward(self, input: torch.Tensor, target: torch.Tensor, cell, ptr):
        min_dist = self.minimum_image_convention_torch(input, target, cell, ptr)
        dists = torch.norm(min_dist, p=2, dim=-1)
        if self.reduction == "mean":
            return torch.mean(dists)
        elif self.reduction == "sum":
            return torch.sum(dists)


class AtomwiseL2Loss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: torch.Tensor,
    ):
        assert natoms.shape[0] == input.shape[0] == target.shape[0]
        assert len(natoms.shape) == 1  # (nAtoms, )

        dists = torch.norm(input - target, p=2, dim=-1)
        loss = natoms * dists

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)


class DDPLoss(nn.Module):
    def __init__(self, loss_fn, reduction: str = "mean") -> None:
        super().__init__()
        self.loss_fn = loss_fn
        self.loss_fn.reduction = "sum"
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        cell: torch.Tensor = None,
        ptr: torch.Tensor = None
    ):
        # zero out nans, if any
        found_nans_or_infs = not torch.all(input.isfinite())
        if found_nans_or_infs is True:
            logging.warning("Found nans while computing loss")
            input = torch.nan_to_num(input, nan=0.0)
        # 修改
        if self.loss_fn.__class__.__name__ == 'MinDist':
            loss = self.loss_fn(input, target, cell, ptr)
        elif natoms is None:
            loss = self.loss_fn(input, target)
        else:  # atom-wise loss
            loss = self.loss_fn(input, target, natoms)
        if self.reduction == "mean":
            num_samples = (
                batch_size if batch_size is not None else input.shape[0]
            )
            num_samples = distutils.all_reduce(
                num_samples, device=input.device
            )
            # Multiply by world size since gradients are averaged
            # across DDP replicas
            return loss * distutils.get_world_size() / num_samples
        else:
            return loss
