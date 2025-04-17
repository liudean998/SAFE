"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from typing import Optional

import torch

from ocpmodels.common.utils import scatter_det
from ocpmodels.modules.scaling import ScaleFactor

from .base_layers import Dense, ResidualLayer


class AtomUpdateBlock(torch.nn.Module):
    """
    Aggregate the message embeddings of the atoms

    Arguments
    ---------
    emb_size_atom: int
        Embedding size of the atoms.
    emb_size_edge: int
        Embedding size of the edges.
    emb_size_rbf: int
        Embedding size of the radial basis.
    nHidden: int
        Number of residual blocks.
    activation: callable/str
        Name of the activation function to use in the dense layers.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        nHidden: int,
        activation=None,
    ) -> None:
        super().__init__()

        self.dense_rbf = Dense(
            emb_size_rbf, emb_size_edge, activation=None, bias=False
        )
        self.scale_sum = ScaleFactor()

        self.layers = self.get_mlp(
            emb_size_edge, emb_size_atom, nHidden, activation
        )

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

    def forward(self, h: torch.Tensor, m, basis_rad, idx_atom):
        """
        Returns
        -------
        h: torch.Tensor, shape=(nAtoms, emb_size_atom)
            Atom embedding.
        """
        nAtoms = h.shape[0]

        bases_emb = self.dense_rbf(basis_rad)  # (nEdges, emb_size_edge)
        x = m * bases_emb

        x2 = scatter_det(
            x, idx_atom, dim=0, dim_size=nAtoms, reduce="sum"
        )  # (nAtoms, emb_size_edge)
        x = self.scale_sum(x2, ref=m)

        for layer in self.layers:
            x = layer(x)  # (nAtoms, emb_size_atom)

        return x


class OutputBlock(AtomUpdateBlock):
    """
    Combines the atom update block and subsequent final dense layer.

    Arguments
    ---------
    emb_size_atom: int
        Embedding size of the atoms.
    emb_size_edge: int
        Embedding size of the edges.
    emb_size_rbf: int
        Embedding size of the radial basis.
    nHidden: int
        Number of residual blocks before adding the atom embedding.
    nHidden_afteratom: int
        Number of residual blocks after adding the atom embedding.
    activation: str
        Name of the activation function to use in the dense layers.
    direct_forces: bool
        If true directly predict forces, i.e. without taking the gradient
        of the energy potential.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        nHidden: int,
        nHidden_afteratom: int,
        activation: Optional[str] = None,
        direct_forces: bool = True,
    ) -> None:
        super().__init__(
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            emb_size_rbf=emb_size_rbf,
            nHidden=nHidden,
            activation=activation,
        )

        self.direct_forces = direct_forces

        self.seq_energy_pre = self.layers  # inherited from parent class
        if nHidden_afteratom >= 1:
            self.seq_energy2 = self.get_mlp(
                emb_size_atom, emb_size_atom, nHidden_afteratom, activation
            )
            self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        else:
            self.seq_energy2 = None

        if self.direct_forces:
            self.scale_rbf_F = ScaleFactor()
            self.seq_forces = self.get_mlp(
                emb_size_edge, emb_size_edge, nHidden, activation
            )
            self.dense_rbf_F = Dense(
                emb_size_rbf, emb_size_edge, activation=None, bias=False
            )

    def forward(self, h: torch.Tensor, m: torch.Tensor, basis_rad, idx_atom):
        """
        Returns
        -------
        torch.Tensor, shape=(nAtoms, emb_size_atom)
            Output atom embeddings.
        torch.Tensor, shape=(nEdges, emb_size_edge)
            Output edge embeddings.
        """
        nAtoms = h.shape[0]

        # ------------------------ Atom embeddings ------------------------ #
        basis_emb_E = self.dense_rbf(basis_rad)  # (nEdges, emb_size_edge)
        x = m * basis_emb_E

        x_E = scatter_det(
            x, idx_atom, dim=0, dim_size=nAtoms, reduce="sum"
        )  # (nAtoms, emb_size_edge)
        x_E = self.scale_sum(x_E, ref=m)

        for layer in self.seq_energy_pre:
            x_E = layer(x_E)  # (nAtoms, emb_size_atom)

        if self.seq_energy2 is not None:
            x_E = x_E + h
            x_E = x_E * self.inv_sqrt_2
            for layer in self.seq_energy2:
                x_E = layer(x_E)  # (nAtoms, emb_size_atom)

        # ------------------------- Edge embeddings ------------------------ #
        if self.direct_forces:
            x_F = m
            for _, layer in enumerate(self.seq_forces):
                x_F = layer(x_F)  # (nEdges, emb_size_edge)

            basis_emb_F = self.dense_rbf_F(basis_rad)
            # (nEdges, emb_size_edge)
            x_F_basis = x_F * basis_emb_F
            x_F = self.scale_rbf_F(x_F_basis, ref=x_F)
        else:
            x_F = 0
        # ------------------------------------------------------------------ #

        return x_E, x_F


class OutputBlockStru(AtomUpdateBlock):
    """
    Combines the atom update block and subsequent final dense layer.

    Arguments
    ---------
    emb_size_atom: int
        Embedding size of the atoms.
    emb_size_edge: int
        Embedding size of the edges.
    emb_size_rbf: int
        Embedding size of the radial basis.
    nHidden: int
        Number of residual blocks before adding the atom embedding.
    nHidden_afteratom: int
        Number of residual blocks after adding the atom embedding.
    activation: str
        Name of the activation function to use in the dense layers.
    direct_forces: bool
        If true directly predict forces, i.e. without taking the gradient
        of the energy potential.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        nHidden: int,
        nHidden_afteratom: int,
        activation: Optional[str] = None,
        regress_forces: bool = True,
        direct_forces: bool = True,
        update_v: bool = False, # 修改
        update_p: bool = False # 修改
    ) -> None:
        super().__init__(
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            emb_size_rbf=emb_size_rbf,
            nHidden=nHidden,
            activation=activation,
        )
        self.regress_forces = regress_forces
        self.direct_forces = direct_forces
        self.update_v = update_v
        self.update_p = update_p

        self.seq_energy_pre = self.layers  # inherited from parent class
        if nHidden_afteratom >= 1:
            self.seq_energy2 = self.get_mlp(
                emb_size_atom, emb_size_atom, nHidden_afteratom, activation
            )
            self.inv_sqrt_2 = 1 / math.sqrt(2.0)
            if self.update_p:
                # self.seq_pos2 = self.seq_energy2 = self.get_mlp(
                #     emb_size_atom, emb_size_atom, nHidden_afteratom, activation)
                self.seq_pos2 = self.get_mlp(
                    emb_size_atom, emb_size_atom, nHidden_afteratom, activation)
        else:
            self.seq_energy2 = None
            self.seq_pos2 = None

        if self.regress_forces and self.direct_forces:
            self.scale_rbf_F = ScaleFactor()
            self.seq_forces = self.get_mlp(
                emb_size_edge, emb_size_edge, nHidden, activation
            )
            self.dense_rbf_F = Dense(
                emb_size_rbf, emb_size_edge, activation=None, bias=False
            )
        if self.update_v:
            # self.scale_rbf_V = ScaleFactor()
            self.seq_vec = self.get_mlp(emb_size_edge, emb_size_edge,
                                           nHidden, activation)
            self.dense_rbf_V = Dense(
                emb_size_rbf, emb_size_edge, activation=None, bias=False
            )

        if self.update_p:
            self.dense_rbf_P = Dense(
                emb_size_rbf, emb_size_edge, activation=None, bias=False
            )
            self.seq_pos_pre = self.get_mlp(
                emb_size_edge, emb_size_atom, nHidden, activation
            )

    def forward(self,
                h: torch.Tensor,
                m: torch.Tensor,
                basis_rad,
                idx_atom,
                ):
        """
        Returns
        -------
        torch.Tensor, shape=(nAtoms, emb_size_atom)
            Output atom embeddings.
        torch.Tensor, shape=(nEdges, emb_size_edge)
            Output edge embeddings.
        """
        nAtoms = h.shape[0]
        #####
        #能量
        #####
        basis_emb_E = self.dense_rbf(basis_rad.clone())  # (nEdges, emb_size_edge)
        x = m * basis_emb_E
        x_E = scatter_det(
            x, idx_atom, dim=0, dim_size=nAtoms, reduce="sum"
        )  # (nAtoms, emb_size_edge)
        x_E = self.scale_sum(x_E, ref=m)
            # print(x_E.shape)
        for layer in self.seq_energy_pre:
            x_E = layer(x_E)  # (nAtoms, emb_size_atom)
        if self.seq_energy2 is not None:
            x_E = x_E + h
            x_E = x_E * self.inv_sqrt_2
            for layer in self.seq_energy2:
                x_E = layer(x_E)  # (nAtoms, emb_size_atom)
        #####
        #向量
        #####
        if self.update_v:
            x_V = m.clone()
            for _, layer in enumerate(self.seq_vec):
                x_V = layer(x_V)
            basis_emb_V = self.dense_rbf_V(basis_rad.clone())
            x_V = x_V * basis_emb_V # (nEdges, emb_size_edge)
        else:
            x_V = None

        #####
        #坐标
        #####
        if  self.update_p:
            basis_emb_P = self.dense_rbf_P(basis_rad.clone())
            p = m * basis_emb_P
            x_P = scatter_det(p, idx_atom,
                              dim=0, dim_size=nAtoms, reduce="sum")
            for layer in self.seq_pos_pre:
                x_P = layer(x_P)
            if self.seq_pos2:
                x_P = x_P + h
                for layer in self.seq_pos2:
                    x_P = layer(x_P)
        else:
            x_P = None

        #####
        #力
        #####
        if self.regress_forces and self.direct_forces:
            x_F = m
            for _, layer in enumerate(self.seq_forces):
                x_F = layer(x_F)  # (nEdges, emb_size_edge)
            basis_emb_F = self.dense_rbf_F(basis_rad)
            # (nEdges, emb_size_edge)
            x_F_basis = x_F * basis_emb_F
            x_F = self.scale_rbf_F(x_F_basis, ref=x_F)
        else:
            x_F = None

        return x_E, x_V, x_P, x_F

class OutputBlockStru1(AtomUpdateBlock):
    """
    Combines the atom update block and subsequent final dense layer.

    Arguments
    ---------
    emb_size_atom: int
        Embedding size of the atoms.
    emb_size_edge: int
        Embedding size of the edges.
    emb_size_rbf: int
        Embedding size of the radial basis.
    nHidden: int
        Number of residual blocks before adding the atom embedding.
    nHidden_afteratom: int
        Number of residual blocks after adding the atom embedding.
    activation: str
        Name of the activation function to use in the dense layers.
    direct_forces: bool
        If true directly predict forces, i.e. without taking the gradient
        of the energy potential.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        nHidden: int,
        nHidden_afteratom: int,
        activation: Optional[str] = None,
        direct_forces: bool = True,
        update_v: bool = False, # 修改
        update_p: bool = False # 修改
    ) -> None:
        super().__init__(
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            emb_size_rbf=emb_size_rbf,
            nHidden=nHidden,
            activation=activation,
        )

        self.direct_forces = direct_forces
        self.update_v = update_v
        self.update_p = update_p

        self.seq_energy_pre = self.layers  # inherited from parent class
        if nHidden_afteratom >= 1:
            self.seq_energy2 = self.get_mlp(
                emb_size_atom, emb_size_atom, nHidden_afteratom, activation
            )
            self.inv_sqrt_2 = 1 / math.sqrt(2.0)
            if self.update_p:
                self.seq_pos2 = self.seq_energy2 = self.get_mlp(
                    emb_size_atom, emb_size_atom, nHidden_afteratom, activation)
        else:
            self.seq_energy2 = None
            self.seq_pos2 = None

        if self.direct_forces:
            self.scale_rbf_F = ScaleFactor()
            self.seq_forces = self.get_mlp(
                emb_size_edge, emb_size_edge, nHidden, activation
            )
            self.dense_rbf_F = Dense(
                emb_size_rbf, emb_size_edge, activation=None, bias=False
            )
        if self.update_v:
            # self.scale_rbf_V = ScaleFactor()
            self.seq_vec = self.get_mlp(emb_size_edge, emb_size_edge,
                                           nHidden, activation)
            self.dense_rbf_V = Dense(
                emb_size_rbf, emb_size_edge, activation=None, bias=False
            )
            self.seq_d = self.get_mlp(emb_size_edge, emb_size_edge,
                                        nHidden, activation)
            self.dense_rbf_D = Dense(
                emb_size_rbf, emb_size_edge, activation=None, bias=False
            )

        if self.update_p:
            self.dense_rbf_P = Dense(
                emb_size_rbf, emb_size_edge, activation=None, bias=False
            )
            self.seq_pos_pre = self.get_mlp(
                emb_size_edge, emb_size_atom, nHidden, activation
            )

    def forward(self,
                h: torch.Tensor,
                m: torch.Tensor,
                basis_rad,
                idx_atom,
                out_energy=True,
                out_vector=False,
                out_pos=False
                ):
        """
        Returns
        -------
        torch.Tensor, shape=(nAtoms, emb_size_atom)
            Output atom embeddings.
        torch.Tensor, shape=(nEdges, emb_size_edge)
            Output edge embeddings.
        """
        ######
        #能量
        ######
        if out_energy: # 修改
            nAtoms = h.shape[0]
            # ------------------------ 能量 ------------------------ #
            basis_emb_E = self.dense_rbf(basis_rad.clone())  # (nEdges, emb_size_edge)
            x = m * basis_emb_E

            x_E = scatter_det(
                x, idx_atom, dim=0, dim_size=nAtoms, reduce="sum"
            )  # (nAtoms, emb_size_edge)
            x_E = self.scale_sum(x_E, ref=m)
                # print(x_E.shape)
            for layer in self.seq_energy_pre:
                x_E = layer(x_E)  # (nAtoms, emb_size_atom)
            if self.seq_energy2 is not None:
                x_E = x_E + h
                x_E = x_E * self.inv_sqrt_2
                for layer in self.seq_energy2:
                    x_E = layer(x_E)  # (nAtoms, emb_size_atom)
        else:
            x_E = None

        ######
        # 向量
        ######
        if out_vector:
            x_V = m.clone()
            for _, layer in enumerate(self.seq_vec):
                x_V = layer(x_V)
            basis_emb_V = self.dense_rbf_V(basis_rad.clone())
            x_V = x_V * basis_emb_V # (nEdges, emb_size_edge)
            x_D = m.clone()
            for _, layer in enumerate(self.seq_d):
                x_D = layer(x_D)
            basis_emb_D = self.dense_rbf_D(basis_rad.clone())
            x_D = x_D*basis_emb_D
        else:
            x_V = None
            x_D = None

        ######
        #坐标
        ######
        if  out_pos:
            nAtoms = h.shape[0]
            basis_emb_P = self.dense_rbf_P(basis_rad.clone())
            p = m * basis_emb_P
            x_P = scatter_det(p, idx_atom,
                              dim=0, dim_size=nAtoms, reduce="sum")
            for layer in self.seq_pos_pre:
                x_P = layer(x_P)
            if self.seq_pos2:
                x_P = x_P + h
                for layer in self.seq_pos2:
                    x_P = layer(x_P)
        else:
            x_P = None

        return x_E, x_V, x_P, x_D

class OutputBlockStruForce1(AtomUpdateBlock):
    """
    如果共享层，效果更好，由于坐标和力都是和原子对应的，直接使用energy输出的x_E作为后续
    多层感知机的输出
    Combines the atom update block and subsequent final dense layer.

    Arguments
    ---------
    emb_size_atom: int
        Embedding size of the atoms.
    emb_size_edge: int
        Embedding size of the edges.
    emb_size_rbf: int
        Embedding size of the radial basis.
    nHidden: int
        Number of residual blocks before adding the atom embedding.
    nHidden_afteratom: int
        Number of residual blocks after adding the atom embedding.
    activation: str
        Name of the activation function to use in the dense layers.
    direct_forces: bool
        If true directly predict forces, i.e. without taking the gradient
        of the energy potential.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        nHidden: int,
        nHidden_afteratom: int,
        activation: Optional[str] = None,
        regress_forces: bool = True,
        direct_forces: bool = True,
        update_v: bool = False, # 修改
        update_p: bool = False # 修改
    ) -> None:
        super().__init__(
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            emb_size_rbf=emb_size_rbf,
            nHidden=nHidden,
            activation=activation,
        )
        self.regress_forces = regress_forces
        self.direct_forces = direct_forces
        self.update_v = update_v
        self.update_p = update_p

        self.seq_energy_pre = self.layers  # inherited from parent class
        if nHidden_afteratom >= 1:
            self.seq_energy2 = self.get_mlp(
                emb_size_atom, emb_size_atom, nHidden_afteratom, activation
            )
            self.inv_sqrt_2 = 1 / math.sqrt(2.0)

        else:
            self.seq_energy2 = None

        if self.update_v:
            # self.scale_rbf_V = ScaleFactor()
            self.seq_vec = self.get_mlp(emb_size_edge, emb_size_edge,
                                           nHidden, activation)
            # self.dense_rbf_V = Dense(
            #     emb_size_rbf, emb_size_edge, activation=None, bias=False
            # )

    def forward(self,
                h: torch.Tensor,
                m: torch.Tensor,
                basis_rad,
                idx_atom,
                ):
        """
        Returns
        -------
        torch.Tensor, shape=(nAtoms, emb_size_atom)
            Output atom embeddings.
        torch.Tensor, shape=(nEdges, emb_size_edge)
            Output edge embeddings.
        """
        nAtoms = h.shape[0]
        # ------------------------ 能量 ------------------------ #
        basis_emb_E = self.dense_rbf(basis_rad.clone())  # (nEdges, emb_size_edge)
        x = m * basis_emb_E
        x_E = scatter_det(
            x, idx_atom, dim=0, dim_size=nAtoms, reduce="sum"
        )  # (nAtoms, emb_size_edge)
        x_E = self.scale_sum(x_E, ref=m)
            # print(x_E.shape)
        for layer in self.seq_energy_pre:
            x_E = layer(x_E)  # (nAtoms, emb_size_atom)
        if self.seq_energy2 is not None:
            x_E = x_E + h
            x_E = x_E * self.inv_sqrt_2
            for layer in self.seq_energy2:
                x_E = layer(x_E)  # (nAtoms, emb_size_atom)
        # -----------------------向量 自添加------------------------------ #
        if self.update_v:
            # x_V = m.clone()
            # for _, layer in enumerate(self.seq_vec):
            #     # x_V = layer(x_V)
            #     x_V = layer(x) # 共享1
            x_V = x # 共享2
            # basis_emb_V = self.dense_rbf_V(basis_rad.clone())
            # x_V = x_V * basis_emb_E # (nEdges, emb_size_edge)
        else:
            x_V = None
        return x_E, x_V

class OutputBlockMask(AtomUpdateBlock):
    """
    Combines the atom update block and subsequent final dense layer.

    Arguments
    ---------
    emb_size_atom: int
        Embedding size of the atoms.
    emb_size_edge: int
        Embedding size of the edges.
    emb_size_rbf: int
        Embedding size of the radial basis.
    nHidden: int
        Number of residual blocks before adding the atom embedding.
    nHidden_afteratom: int
        Number of residual blocks after adding the atom embedding.
    activation: str
        Name of the activation function to use in the dense layers.
    direct_forces: bool
        If true directly predict forces, i.e. without taking the gradient
        of the energy potential.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        nHidden: int,
        nHidden_afteratom: int,
        activation: Optional[str] = None,
        direct_forces: bool = True,
        update_v: bool = False # 修改
    ) -> None:
        super().__init__(
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            emb_size_rbf=emb_size_rbf,
            nHidden=nHidden,
            activation=activation,
        )

        self.direct_forces = direct_forces
        self.update_v = update_v

        self.seq_energy_pre = self.layers  # inherited from parent class
        if nHidden_afteratom >= 1:
            self.seq_energy2 = self.get_mlp(
                emb_size_atom, emb_size_atom, nHidden_afteratom, activation
            )
            self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        else:
            self.seq_energy2 = None

        if self.direct_forces:
            self.scale_rbf_F = ScaleFactor()
            self.seq_forces = self.get_mlp(
                emb_size_edge, emb_size_edge, nHidden, activation
            )
            self.dense_rbf_F = Dense(
                emb_size_rbf, emb_size_edge, activation=None, bias=False
            )
        if self.update_v:
            # self.scale_rbf_V = ScaleFactor()
            self.seq_vec = self.get_mlp(emb_size_edge, emb_size_edge,
                                           nHidden, activation)
            self.dense_rbf_V = Dense(
                emb_size_rbf, emb_size_edge, activation=None, bias=False
            )

    def forward(self,
                h: torch.Tensor,
                m: torch.Tensor,
                basis_rad,
                idx_atom,
                mask=None,
                tags_mask=None,
                out_energy=True,
                out_vector=False
                ):
        """
        Returns
        -------
        torch.Tensor, shape=(nAtoms, emb_size_atom)
            Output atom embeddings.
        torch.Tensor, shape=(nEdges, emb_size_edge)
            Output edge embeddings.
        """
        if mask is not None: # 修改
            m = m[mask]
            basis_rad = basis_rad[mask]
            idx_atom = idx_atom[mask]
        if out_energy: # 修改
            nAtoms = h.shape[0]
            # ------------------------ 能量 ------------------------ #
            basis_emb_E = self.dense_rbf(basis_rad)  # (nEdges, emb_size_edge)
            x = m * basis_emb_E

            x_E = scatter_det(
                x, idx_atom, dim=0, dim_size=nAtoms, reduce="sum"
            )  # (nAtoms, emb_size_edge)
            x_E = self.scale_sum(x_E, ref=m)
            if tags_mask is not None:
                x_E = x_E[tags_mask]
                h = h[tags_mask]
                # print(x_E.shape)
            for layer in self.seq_energy_pre:
                x_E = layer(x_E)  # (nAtoms, emb_size_atom)
            if self.seq_energy2 is not None:
                x_E = x_E + h
                x_E = x_E * self.inv_sqrt_2
                for layer in self.seq_energy2:
                    x_E = layer(x_E)  # (nAtoms, emb_size_atom)
        else:
            x_E = 0

        # ------------------------- 力 ------------------------ #
        if self.direct_forces:
            x_F = m.clone()
            for _, layer in enumerate(self.seq_forces):
                x_F = layer(x_F)  # (nEdges, emb_size_edge)

            basis_emb_F = self.dense_rbf_F(basis_rad)
            # (nEdges, emb_size_edge)
            x_F_basis = x_F * basis_emb_F
            x_F = self.scale_rbf_F(x_F_basis, ref=x_F)
        else:
            x_F = 0
        # -----------------------坐标 自添加------------------------------ #
        if self.update_v and out_vector:
            x_V = m.clone()
            for _, layer in enumerate(self.seq_vec):
                x_V = layer(x_V)
            basis_emb_V = self.dense_rbf_V(basis_rad)
            x_V = x_V * basis_emb_V # (nEdges, emb_size_edge)
            # x_V = self.scale_rbf_V(x_V_basis, ref=x_V)
            return x_E, x_F, x_V
        else:
            return x_E, x_F


class OutputBlockStruH(AtomUpdateBlock):
    """
    Combines the atom update block and subsequent final dense layer.

    Arguments
    ---------
    emb_size_atom: int
        Embedding size of the atoms.
    emb_size_edge: int
        Embedding size of the edges.
    emb_size_rbf: int
        Embedding size of the radial basis.
    nHidden: int
        Number of residual blocks before adding the atom embedding.
    nHidden_afteratom: int
        Number of residual blocks after adding the atom embedding.
    activation: str
        Name of the activation function to use in the dense layers.
    direct_forces: bool
        If true directly predict forces, i.e. without taking the gradient
        of the energy potential.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        nHidden: int,
        nHidden_afteratom: int,
        activation: Optional[str] = None,
        direct_forces: bool = True,
        update_v: bool = False, # 修改
        update_p: bool = False # 修改
    ) -> None:
        super().__init__(
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            emb_size_rbf=emb_size_rbf,
            nHidden=nHidden,
            activation=activation,
        )

        self.direct_forces = direct_forces
        self.update_v = update_v
        self.update_p = update_p

        self.seq_energy_pre = self.layers  # inherited from parent class
        if nHidden_afteratom >= 1:
            self.seq_energy2 = self.get_mlp(
                emb_size_atom, emb_size_atom, nHidden_afteratom, activation
            )
            self.inv_sqrt_2 = 1 / math.sqrt(2.0)
            if self.update_p:
                self.seq_pos2 = self.seq_energy2 = self.get_mlp(
                    emb_size_atom, emb_size_atom, nHidden_afteratom, activation)
        else:
            self.seq_energy2 = None
            self.seq_pos2 = None

        if self.direct_forces:
            self.scale_rbf_F = ScaleFactor()
            self.seq_forces = self.get_mlp(
                emb_size_edge, emb_size_edge, nHidden, activation
            )
            self.dense_rbf_F = Dense(
                emb_size_rbf, emb_size_edge, activation=None, bias=False
            )
        if self.update_v:
            # self.scale_rbf_V = ScaleFactor()
            self.seq_vec = self.get_mlp(emb_size_edge, emb_size_edge,
                                           nHidden, activation)
            self.dense_rbf_V = Dense(
                emb_size_rbf, emb_size_edge, activation=None, bias=False
            )

        if self.update_p:
            self.dense_rbf_P = Dense(
                emb_size_rbf, emb_size_edge, activation=None, bias=False
            )
            self.seq_pos_pre = self.get_mlp(
                emb_size_edge, emb_size_atom, nHidden, activation
            )

    def forward(self,
                h: torch.Tensor,
                m: torch.Tensor,
                basis_rad,
                idx_atom,
                out_energy=True,
                out_vector=False,
                out_pos=False
                ):
        """
        Returns
        -------
        torch.Tensor, shape=(nAtoms, emb_size_atom)
            Output atom embeddings.
        torch.Tensor, shape=(nEdges, emb_size_edge)
            Output edge embeddings.
        """
        if out_energy: # 修改
            nAtoms = h.shape[0]
            # ------------------------ 能量 ------------------------ #
            basis_emb_E = self.dense_rbf(basis_rad.clone())  # (nEdges, emb_size_edge)
            x_h = m * basis_emb_E
            for layer in self.seq_pos_pre:
                x_h = layer(x_h) # 边对应的能量
            # x_E = scatter_det(
            #     x, idx_atom, dim=0, dim_size=nAtoms, reduce="sum"
            # )  # (nAtoms, emb_size_edge)
            # x_E = self.scale_sum(x_E, ref=m)
                # print(x_E.shape)
            # for layer in self.seq_energy_pre:
            #       # (nAtoms, emb_size_atom)
            if self.seq_energy2 is not None:
                # x_E = x_E + h
                x_E = h * self.inv_sqrt_2 # 点对应的能量
                for layer in self.seq_energy2:
                    x_E = layer(x_E)  # (nAtoms, emb_size_atom)
            else:
                x_E = None
        else:
            x_E = None
            x_h = None

        # -----------------------向量 自添加------------------------------ #
        if out_vector:
            x_V = m.clone()
            for _, layer in enumerate(self.seq_vec):
                x_V = layer(x_V)
            basis_emb_V = self.dense_rbf_V(basis_rad.clone())
            x_V = x_V * basis_emb_V # (nEdges, emb_size_edge)
        else:
            x_V = None

        # -----------------------向量 自添加------------------------------ #
        if  out_pos:
            nAtoms = h.shape[0]
            basis_emb_P = self.dense_rbf_P(basis_rad.clone())
            p = m * basis_emb_P
            x_P = scatter_det(p, idx_atom,
                              dim=0, dim_size=nAtoms, reduce="sum")
            for layer in self.seq_pos_pre:
                x_P = layer(x_P)
            if self.seq_pos2:
                x_P = x_P + h
                for layer in self.seq_pos2:
                    x_P = layer(x_P)
        else:
            x_P = None

        return x_E,x_h, x_V, x_P