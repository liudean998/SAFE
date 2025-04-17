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

class MaeV(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(self, input: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        diff = torch.abs(input-target)
        if weight is not None:
            diff = diff * weight.view(-1, 1)
        if self.reduction == "mean":
            return torch.mean(diff)
        elif self.reduction == "sum":
            diff = torch.sum(diff, dim=1)
            return torch.sum(diff)# 修改逻辑，先对每行求和，再对所有行求平均值

class MseV(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        diff = (input-target)**2
        return torch.mean(diff)
        # if self.reduction == "mean":
        #     return torch.mean(diff)
        # elif self.reduction == "sum":
        #     return torch.sum(diff)

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
    def __init__(self, loss_fn, reduction: str = "mean", loss_name: str = '') -> None:
        super().__init__()
        self.loss_fn = loss_fn
        self.loss_fn.reduction = "sum"
        self.reduction = reduction
        self.loss_name = loss_name # 修改-增加 对原结构无影响
        assert reduction in ["mean", "sum"]

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        weight: Optional[torch.Tensor] = None
    ):
        # zero out nans, if any
        found_nans_or_infs = not torch.all(input.isfinite())
        if found_nans_or_infs is True:
            logging.warning("Found nans while computing loss")
            input = torch.nan_to_num(input, nan=0.0)

        if natoms is None:
            if weight is not None:
                loss = self.loss_fn(input, target, weight=weight)
            else:
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
