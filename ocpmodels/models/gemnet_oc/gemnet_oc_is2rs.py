"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
import time
from typing import Dict, Optional, Union

import torch

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    conditional_grad,
    scatter_det,
)

from ocpmodels.models.gemnet_oc.gemnet_oc import GemNetOC
from ocpmodels.modules.scaling.compat import load_scales_compat
from .initializers import get_initializer
from .interaction_indices import (
    get_mixed_triplets,
    get_quadruplets,
    get_triplets,
)
from .layers.atom_update_block import  OutputBlockStru, OutputBlock
from .layers.base_layers import Dense, ResidualLayer
from .layers.embedding_block import AtomEmbedding, EdgeEmbedding
from .layers.force_scaler import ForceScaler
from .layers.interaction_block import InteractionBlock
from .layers.radial_basis import RadialBasisVec, RadialBasis
from .layers.spherical_basis import  CircularBasisLayerVec, SphericalBasisLayerVec, CircularBasisLayer, SphericalBasisLayer
from .utils import (
    get_inner_idx,
    inner_product_clamped,
)

@registry.register_model("gemnet_is2rs")
class GemNetOCRS(GemNetOC):
    def __init__(
            self,
            num_atoms: Optional[int],
            bond_feat_dim: int,
            num_targets: int,
            num_spherical: int,
            num_radial: int,
            num_blocks: int,
            emb_size_atom: int,
            emb_size_edge: int,
            emb_size_trip_in: int,
            emb_size_trip_out: int,
            emb_size_quad_in: int,
            emb_size_quad_out: int,
            emb_size_aint_in: int,
            emb_size_aint_out: int,
            emb_size_rbf: int,
            emb_size_cbf: int,
            emb_size_sbf: int,
            num_before_skip: int,
            num_after_skip: int,
            num_concat: int,
            num_atom: int,
            num_output_afteratom: int,
            num_atom_emb_layers: int = 0,
            num_global_out_layers: int = 2,
            regress_forces: bool = True,
            direct_forces: bool = False,
            use_pbc: bool = True,
            scale_backprop_forces: bool = False, # 保守力场设置，False时，计算力为能量的负梯度
            cutoff: float = 6.0,
            cutoff_qint: Optional[float] = None,
            cutoff_aeaint: Optional[float] = None,
            cutoff_aint: Optional[float] = None,
            max_neighbors: int = 50,
            max_neighbors_qint: Optional[int] = None,
            max_neighbors_aeaint: Optional[int] = None,
            max_neighbors_aint: Optional[int] = None,
            enforce_max_neighbors_strictly: bool = True,
            rbf: Dict[str, str] = {"name": "gaussian"},
            rbf_spherical: Optional[dict] = None,
            envelope: Dict[str, Union[str, int]] = {
                "name": "polynomial",
                "exponent": 5,
            },
            cbf: Dict[str, str] = {"name": "spherical_harmonics"},
            sbf: Dict[str, str] = {"name": "spherical_harmonics"},
            extensive: bool = True,
            forces_coupled: bool = False,
            output_init: str = "HeOrthogonal",
            activation: str = "silu",
            quad_interaction: bool = False,
            atom_edge_interaction: bool = False,
            edge_atom_interaction: bool = False,
            atom_interaction: bool = False,
            scale_basis: bool = False,
            qint_tags: list = None,
            num_elements: int = 83,
            otf_graph: bool = False,
            scale_file: Optional[str] = None,
            # 训练pos的参数
            update_v: bool = True,
            update_p: bool = True,
            edge_mask: bool = False,
            edge_mask_tags=0,
            **kwargs,  # backwards compatibility with deprecated arguments
    ) -> None:
        super().__init__(num_atoms, bond_feat_dim, num_targets, num_spherical, num_radial, num_blocks,
                         emb_size_atom, emb_size_edge, emb_size_trip_in, emb_size_trip_out, emb_size_quad_in,
                         emb_size_quad_out, emb_size_aint_in, emb_size_aint_out, emb_size_rbf, emb_size_cbf,
                         emb_size_sbf, num_before_skip, num_after_skip, num_concat, num_atom, num_output_afteratom,
                         num_atom_emb_layers, num_global_out_layers, regress_forces, direct_forces, use_pbc,
                         scale_backprop_forces,cutoff, cutoff_qint, cutoff_aeaint, cutoff_aint, max_neighbors,
                         max_neighbors_qint, max_neighbors_aeaint, max_neighbors_aint, enforce_max_neighbors_strictly,
                         rbf, rbf_spherical, envelope, cbf, sbf, extensive, forces_coupled, output_init, activation,
                         quad_interaction, atom_edge_interaction, edge_atom_interaction, atom_interaction, scale_basis,
                         qint_tags, num_elements, otf_graph, # scale_file,
         )
        if len(kwargs) > 0:
            logging.warning(f"Unrecognized arguments: {list(kwargs.keys())}")
        self.num_targets = num_targets
        assert num_blocks > 0
        self.num_blocks = num_blocks
        self.extensive = extensive

        self.atom_edge_interaction = atom_edge_interaction
        self.edge_atom_interaction = edge_atom_interaction
        self.atom_interaction = atom_interaction
        self.quad_interaction = quad_interaction
        self.qint_tags = torch.tensor(qint_tags)
        self.otf_graph = otf_graph
        if not rbf_spherical:
            rbf_spherical = rbf

        self.set_cutoffs(cutoff, cutoff_qint, cutoff_aeaint, cutoff_aint)
        self.set_max_neighbors(
            max_neighbors,
            max_neighbors_qint,
            max_neighbors_aeaint,
            max_neighbors_aint,
        )
        self.enforce_max_neighbors_strictly = enforce_max_neighbors_strictly
        self.use_pbc = use_pbc

        self.direct_forces = direct_forces
        self.forces_coupled = forces_coupled
        self.regress_forces = regress_forces
        self.force_scaler = ForceScaler(enabled=scale_backprop_forces)

        self.update_v = update_v
        self.update_p = update_p

        self.init_basis_functions(
            num_radial,
            num_spherical,
            rbf,
            rbf_spherical,
            envelope,
            cbf,
            sbf,
            scale_basis,
        )
        self.init_shared_basis_layers(
            num_radial, num_spherical, emb_size_rbf, emb_size_cbf, emb_size_sbf
        )

        # Embedding blocks
        self.atom_emb = AtomEmbedding(emb_size_atom, num_elements)
        self.edge_emb = EdgeEmbedding(
            emb_size_atom, num_radial, emb_size_edge, activation=activation
        )

        # Interaction Blocks
        int_blocks = []
        for _ in range(num_blocks):
            int_blocks.append(
                InteractionBlock(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_trip_in=emb_size_trip_in,
                    emb_size_trip_out=emb_size_trip_out,
                    emb_size_quad_in=emb_size_quad_in,
                    emb_size_quad_out=emb_size_quad_out,
                    emb_size_a2a_in=emb_size_aint_in,
                    emb_size_a2a_out=emb_size_aint_out,
                    emb_size_rbf=emb_size_rbf,
                    emb_size_cbf=emb_size_cbf,
                    emb_size_sbf=emb_size_sbf,
                    num_before_skip=num_before_skip,
                    num_after_skip=num_after_skip,
                    num_concat=num_concat,
                    num_atom=num_atom,
                    num_atom_emb_layers=num_atom_emb_layers,
                    quad_interaction=quad_interaction,
                    atom_edge_interaction=atom_edge_interaction,
                    edge_atom_interaction=edge_atom_interaction,
                    atom_interaction=atom_interaction,
                    activation=activation,
                )
            )
        self.int_blocks = torch.nn.ModuleList(int_blocks)

        out_blocks = []
        for _ in range(num_blocks + 1):
            out_blocks.append(
                OutputBlockStru(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_rbf=emb_size_rbf,
                    nHidden=num_atom,
                    nHidden_afteratom=num_output_afteratom,
                    activation=activation,
                    direct_forces=direct_forces,
                    update_p=True,
                    update_v=update_v
                )
            )
        self.out_blocks = torch.nn.ModuleList(out_blocks)

        if self.update_v:
            out_mlp_V = [
                            Dense(
                                emb_size_edge * (num_blocks + 1),
                                emb_size_edge,
                                activation=activation,
                            )
                        ] + [
                            ResidualLayer(
                                emb_size_edge,
                                activation=activation,
                            )
                            for _ in range(num_global_out_layers)
                        ]
            self.out_mlp_V = torch.nn.Sequential(*out_mlp_V)
            self.out_v = Dense(
                emb_size_edge, 1, bias=False, activation=None
            )
        out_mlp_P = [
                        Dense(
                            emb_size_atom * (num_blocks + 1),
                            emb_size_atom,
                            activation=activation,
                        )
                    ] + [
                        ResidualLayer(
                            emb_size_atom,
                            activation=activation,
                        )
                        for _ in range(num_global_out_layers)
                    ]
        self.out_mlp_P = torch.nn.Sequential(*out_mlp_P)
        self.out_p = Dense(
            emb_size_atom, 3, bias=False, activation=None
        )
        out_initializer = get_initializer(output_init)
        self.out_p.reset_parameters(out_initializer)
        if update_v:
            self.out_v.reset_parameters(out_initializer)
        load_scales_compat(self, scale_file)
        self.edge_mask = edge_mask
        self.edge_mask_tags = edge_mask_tags
        self.main_graph = None
    def init_basis_functions(
            self,
            num_radial,
            num_spherical,
            rbf,
            rbf_spherical,
            envelope,
            cbf,
            sbf,
            scale_basis,
    ):
        # ------------使用单位化的Vetor和distance做rbf嵌入
        self.radial_basis = RadialBasis(
            num_radial=num_radial,
            cutoff=self.cutoff,
            rbf=rbf,
            envelope=envelope,
            scale_basis=scale_basis,
        )  # 修改
        radial_basis_spherical = RadialBasis(
            num_radial=num_radial,
            cutoff=self.cutoff,
            rbf=rbf_spherical,
            envelope=envelope,
            scale_basis=scale_basis,
        )
        if self.quad_interaction:
            radial_basis_spherical_qint = RadialBasis(
                num_radial=num_radial,
                cutoff=self.cutoff_qint,
                rbf=rbf_spherical,
                envelope=envelope,
                scale_basis=scale_basis,
            )
            self.cbf_basis_qint = CircularBasisLayer(
                num_spherical,
                radial_basis=radial_basis_spherical_qint,
                cbf=cbf,
                scale_basis=scale_basis,
            )

            self.sbf_basis_qint = SphericalBasisLayer(
                num_spherical,
                radial_basis=radial_basis_spherical,
                sbf=sbf,
                scale_basis=scale_basis,
            )
        if self.atom_edge_interaction:
            self.radial_basis_aeaint = RadialBasis(
                num_radial=num_radial,
                cutoff=self.cutoff_aeaint,
                rbf=rbf,
                envelope=envelope,
                scale_basis=scale_basis,
            )
            self.cbf_basis_aeint = CircularBasisLayer(
                num_spherical,
                radial_basis=radial_basis_spherical,
                cbf=cbf,
                scale_basis=scale_basis,
            )
        if self.edge_atom_interaction:
            self.radial_basis_aeaint = RadialBasis(
                num_radial=num_radial,
                cutoff=self.cutoff_aeaint,
                rbf=rbf,
                envelope=envelope,
                scale_basis=scale_basis,
            )
            radial_basis_spherical_aeaint = RadialBasis(
                num_radial=num_radial,
                cutoff=self.cutoff_aeaint,
                rbf=rbf_spherical,
                envelope=envelope,
                scale_basis=scale_basis,
            )
            self.cbf_basis_eaint = CircularBasisLayer(
                num_spherical,
                radial_basis=radial_basis_spherical_aeaint,
                cbf=cbf,
                scale_basis=scale_basis,
            )
        if self.atom_interaction:
            self.radial_basis_aint = RadialBasis(
                num_radial=num_radial,
                cutoff=self.cutoff_aint,
                rbf=rbf,
                envelope=envelope,
                scale_basis=scale_basis,
            )

        self.cbf_basis_tint = CircularBasisLayer(
            num_spherical,
            radial_basis=radial_basis_spherical,
            cbf=cbf,
            scale_basis=scale_basis,
        )
    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        atomic_numbers = data.atomic_numbers.long()
        num_atoms = atomic_numbers.shape[0]
        pos = data.pos
        if self.regress_forces and not self.direct_forces:
            pos.requires_grad_(True)
        (
            main_graph,
            a2a_graph,
            a2ee2a_graph,
            qint_graph,
            id_swap,
            trip_idx_e2e,
            trip_idx_a2e,
            trip_idx_e2a,
            quad_idx,
        ) = self.get_graphs_and_indices(data)
        self.main_graph = main_graph
        _, idx_t = main_graph["edge_index"]

        (
            basis_rad_raw,
            basis_atom_update,
            basis_output,
            bases_qint,
            bases_e2e,
            bases_a2e,
            bases_e2a,
            basis_a2a_rad,
        ) = self.get_bases(
            main_graph=main_graph,
            a2a_graph=a2a_graph,
            a2ee2a_graph=a2ee2a_graph,
            qint_graph=qint_graph,
            trip_idx_e2e=trip_idx_e2e,
            trip_idx_a2e=trip_idx_a2e,
            trip_idx_e2a=trip_idx_e2a,
            quad_idx=quad_idx,
            num_atoms=num_atoms,
        )
        # Embedding block
        h = self.atom_emb(atomic_numbers)
        # (nAtoms, emb_size_atom)
        m = self.edge_emb(h, basis_rad_raw, main_graph["edge_index"])

        # (nEdges, emb_size_edge)
        x_R, x_D = self.out_blocks[0](h, m, basis_output, idx_t)
        # print(x_D, '???')
        # (nEdges, emb_size_edge)
        xs_R = [x_R]
        xs_D = [x_D]
        for i in range(self.num_blocks):
            # Interaction block
            h, m = self.int_blocks[i](
                h=h,
                m=m,
                bases_qint=bases_qint,
                bases_e2e=bases_e2e,
                bases_a2e=bases_a2e,
                bases_e2a=bases_e2a,
                basis_a2a_rad=basis_a2a_rad,
                basis_atom_update=basis_atom_update,
                edge_index_main=main_graph["edge_index"],
                a2ee2a_graph=a2ee2a_graph,
                a2a_graph=a2a_graph,
                id_swap=id_swap,
                trip_idx_e2e=trip_idx_e2e,
                trip_idx_a2e=trip_idx_a2e,
                trip_idx_e2a=trip_idx_e2a,
                quad_idx=quad_idx,
            )  # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)

            x_R, x_D = self.out_blocks[i + 1](h, m, basis_output, idx_t)
            xs_R.append(x_R)
            xs_D.append(x_D)
        # 坐标
        x_R = self.out_mlp_P(torch.cat(xs_R, dim=-1))
        R_t = self.out_p(x_R)
        R_t = R_t.squeeze(1)

        if self.update_v:
            x_D = self.out_mlp_V(torch.cat(xs_D, dim=-1))
            D_t = self.out_v(x_D)
            D_t = D_t.squeeze(1)
        else:
            D_t = None
        # print(D_t)
        return R_t, D_t, main_graph


@registry.register_model("gemnet_is2rsv")
class GemNetOCRSV(GemNetOC):
    def __init__(
            self,
            num_atoms: Optional[int],
            bond_feat_dim: int,
            num_targets: int,
            num_spherical: int,
            num_radial: int,
            num_blocks: int,
            emb_size_atom: int,
            emb_size_edge: int,
            emb_size_trip_in: int,
            emb_size_trip_out: int,
            emb_size_quad_in: int,
            emb_size_quad_out: int,
            emb_size_aint_in: int,
            emb_size_aint_out: int,
            emb_size_rbf: int,
            emb_size_cbf: int,
            emb_size_sbf: int,
            num_before_skip: int,
            num_after_skip: int,
            num_concat: int,
            num_atom: int,
            num_output_afteratom: int,
            num_atom_emb_layers: int = 0,
            num_global_out_layers: int = 2,
            regress_forces: bool = True,
            direct_forces: bool = False,
            use_pbc: bool = True,
            scale_backprop_forces: bool = False, # 保守力场设置，False时，计算力为能量的负梯度
            cutoff: float = 6.0,
            cutoff_qint: Optional[float] = None,
            cutoff_aeaint: Optional[float] = None,
            cutoff_aint: Optional[float] = None,
            max_neighbors: int = 50,
            max_neighbors_qint: Optional[int] = None,
            max_neighbors_aeaint: Optional[int] = None,
            max_neighbors_aint: Optional[int] = None,
            enforce_max_neighbors_strictly: bool = True,
            rbf: Dict[str, str] = {"name": "gaussian"},
            rbf_spherical: Optional[dict] = None,
            envelope: Dict[str, Union[str, int]] = {
                "name": "polynomial",
                "exponent": 5,
            },
            cbf: Dict[str, str] = {"name": "spherical_harmonics"},
            sbf: Dict[str, str] = {"name": "spherical_harmonics"},
            extensive: bool = True,
            forces_coupled: bool = False,
            output_init: str = "HeOrthogonal",
            activation: str = "silu",
            quad_interaction: bool = False,
            atom_edge_interaction: bool = False,
            edge_atom_interaction: bool = False,
            atom_interaction: bool = False,
            scale_basis: bool = False,
            qint_tags: list = None,
            num_elements: int = 83,
            otf_graph: bool = False,
            scale_file: Optional[str] = None,
            # 训练pos的参数
            update_v: bool = True,
            update_p: bool = True,
            edge_mask: bool = False,
            edge_mask_tags=0,
            **kwargs,  # backwards compatibility with deprecated arguments
    ) -> None:
        super().__init__(num_atoms, bond_feat_dim, num_targets, num_spherical, num_radial, num_blocks,
                         emb_size_atom, emb_size_edge, emb_size_trip_in, emb_size_trip_out, emb_size_quad_in,
                         emb_size_quad_out, emb_size_aint_in, emb_size_aint_out, emb_size_rbf, emb_size_cbf,
                         emb_size_sbf, num_before_skip, num_after_skip, num_concat, num_atom, num_output_afteratom,
                         num_atom_emb_layers, num_global_out_layers, regress_forces, direct_forces, use_pbc,
                         scale_backprop_forces,cutoff, cutoff_qint, cutoff_aeaint, cutoff_aint, max_neighbors,
                         max_neighbors_qint, max_neighbors_aeaint, max_neighbors_aint, enforce_max_neighbors_strictly,
                         rbf, rbf_spherical, envelope, cbf, sbf, extensive, forces_coupled, output_init, activation,
                         quad_interaction, atom_edge_interaction, edge_atom_interaction, atom_interaction, scale_basis,
                         qint_tags, num_elements, otf_graph, # scale_file,
         )
        if len(kwargs) > 0:
            logging.warning(f"Unrecognized arguments: {list(kwargs.keys())}")
        self.num_targets = num_targets
        assert num_blocks > 0
        self.num_blocks = num_blocks
        self.extensive = extensive

        self.atom_edge_interaction = atom_edge_interaction
        self.edge_atom_interaction = edge_atom_interaction
        self.atom_interaction = atom_interaction
        self.quad_interaction = quad_interaction
        self.qint_tags = torch.tensor(qint_tags)
        self.otf_graph = otf_graph
        if not rbf_spherical:
            rbf_spherical = rbf

        self.set_cutoffs(cutoff, cutoff_qint, cutoff_aeaint, cutoff_aint)
        self.set_max_neighbors(
            max_neighbors,
            max_neighbors_qint,
            max_neighbors_aeaint,
            max_neighbors_aint,
        )
        self.enforce_max_neighbors_strictly = enforce_max_neighbors_strictly
        self.use_pbc = use_pbc

        self.direct_forces = direct_forces
        self.forces_coupled = forces_coupled
        self.regress_forces = regress_forces
        self.force_scaler = ForceScaler(enabled=scale_backprop_forces)

        self.update_v = update_v
        self.update_p = update_p

        self.init_basis_functions(
            num_radial,
            num_spherical,
            rbf,
            rbf_spherical,
            envelope,
            cbf,
            sbf,
            scale_basis,
        )
        self.init_shared_basis_layers(
            num_radial, num_spherical, emb_size_rbf, emb_size_cbf, emb_size_sbf
        )

        # Embedding blocks
        self.atom_emb = AtomEmbedding(emb_size_atom, num_elements)
        self.edge_emb = EdgeEmbedding(
            emb_size_atom, num_radial, emb_size_edge, activation=activation
        )

        # Interaction Blocks
        int_blocks = []
        for _ in range(num_blocks):
            int_blocks.append(
                InteractionBlock(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_trip_in=emb_size_trip_in,
                    emb_size_trip_out=emb_size_trip_out,
                    emb_size_quad_in=emb_size_quad_in,
                    emb_size_quad_out=emb_size_quad_out,
                    emb_size_a2a_in=emb_size_aint_in,
                    emb_size_a2a_out=emb_size_aint_out,
                    emb_size_rbf=emb_size_rbf,
                    emb_size_cbf=emb_size_cbf,
                    emb_size_sbf=emb_size_sbf,
                    num_before_skip=num_before_skip,
                    num_after_skip=num_after_skip,
                    num_concat=num_concat,
                    num_atom=num_atom,
                    num_atom_emb_layers=num_atom_emb_layers,
                    quad_interaction=quad_interaction,
                    atom_edge_interaction=atom_edge_interaction,
                    edge_atom_interaction=edge_atom_interaction,
                    atom_interaction=atom_interaction,
                    activation=activation,
                )
            )
        self.int_blocks = torch.nn.ModuleList(int_blocks)

        out_blocks = []
        for _ in range(num_blocks + 1):
            out_blocks.append(
                OutputBlockStru(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_rbf=emb_size_rbf,
                    nHidden=num_atom,
                    nHidden_afteratom=num_output_afteratom,
                    activation=activation,
                    regress_forces=regress_forces,
                    direct_forces=direct_forces,
                    update_v=update_v,
                    update_p=update_p
                )
            )
        self.out_blocks = torch.nn.ModuleList(out_blocks)

        if self.update_v:
            out_mlp_V = [
                            Dense(
                                emb_size_edge * (num_blocks + 1),
                                emb_size_edge,
                                activation=activation,
                            )
                        ] + [
                            ResidualLayer(
                                emb_size_edge,
                                activation=activation,
                            )
                            for _ in range(num_global_out_layers)
                        ]
            self.out_mlp_V = torch.nn.Sequential(*out_mlp_V)
            self.out_v = Dense(
                emb_size_edge, 3, bias=False, activation=None
            )
        out_mlp_P = [
                        Dense(
                            emb_size_atom * (num_blocks + 1),
                            emb_size_atom,
                            activation=activation,
                        )
                    ] + [
                        ResidualLayer(
                            emb_size_atom,
                            activation=activation,
                        )
                        for _ in range(num_global_out_layers)
                    ]
        self.out_mlp_P = torch.nn.Sequential(*out_mlp_P)
        self.out_p = Dense(
            emb_size_atom, 3, bias=False, activation=None
        )
        out_initializer = get_initializer(output_init)
        self.out_p.reset_parameters(out_initializer)
        if update_v:
            self.out_v.reset_parameters(out_initializer)
        load_scales_compat(self, scale_file)
        self.edge_mask = edge_mask
        self.edge_mask_tags = edge_mask_tags
        self.main_graph = None
    def init_basis_functions(
            self,
            num_radial,
            num_spherical,
            rbf,
            rbf_spherical,
            envelope,
            cbf,
            sbf,
            scale_basis,
    ):
        # ------------使用单位化的Vetor和distance做rbf嵌入
        self.radial_basis = RadialBasisVec(
            num_radial=int(num_radial / 4),
            cutoff=self.cutoff,
            rbf=rbf,
            envelope=envelope,
            scale_basis=scale_basis,
        )  # 修改
        radial_basis_spherical = RadialBasisVec(
            num_radial=int(num_radial / 4),
            cutoff=self.cutoff,
            rbf=rbf_spherical,
            envelope=envelope,
            scale_basis=scale_basis,
        )
        if self.quad_interaction:
            radial_basis_spherical_qint = RadialBasisVec(
                num_radial=int(num_radial / 4),
                cutoff=self.cutoff_qint,
                rbf=rbf_spherical,
                envelope=envelope,
                scale_basis=scale_basis,
            )
            self.cbf_basis_qint = CircularBasisLayerVec(
                num_spherical,
                radial_basis=radial_basis_spherical_qint,
                cbf=cbf,
                scale_basis=scale_basis,
            )

            self.sbf_basis_qint = SphericalBasisLayerVec(
                num_spherical,
                radial_basis=radial_basis_spherical,
                sbf=sbf,
                scale_basis=scale_basis,
            )
        if self.atom_edge_interaction:
            self.radial_basis_aeaint = RadialBasisVec(
                num_radial=int(num_radial / 4),
                cutoff=self.cutoff_aeaint,
                rbf=rbf,
                envelope=envelope,
                scale_basis=scale_basis,
            )
            self.cbf_basis_aeint = CircularBasisLayerVec(
                num_spherical,
                radial_basis=radial_basis_spherical,
                cbf=cbf,
                scale_basis=scale_basis,
            )
        if self.edge_atom_interaction:
            self.radial_basis_aeaint = RadialBasisVec(
                num_radial=int(num_radial / 4),
                cutoff=self.cutoff_aeaint,
                rbf=rbf,
                envelope=envelope,
                scale_basis=scale_basis,
            )
            radial_basis_spherical_aeaint = RadialBasisVec(
                num_radial=int(num_radial / 4),
                cutoff=self.cutoff_aeaint,
                rbf=rbf_spherical,
                envelope=envelope,
                scale_basis=scale_basis,
            )
            self.cbf_basis_eaint = CircularBasisLayerVec(
                num_spherical,
                radial_basis=radial_basis_spherical_aeaint,
                cbf=cbf,
                scale_basis=scale_basis,
            )
        if self.atom_interaction:
            self.radial_basis_aint = RadialBasisVec(
                num_radial=int(num_radial / 4),
                cutoff=self.cutoff_aint,
                rbf=rbf,
                envelope=envelope,
                scale_basis=scale_basis,
            )

        self.cbf_basis_tint = CircularBasisLayerVec(
            num_spherical,
            radial_basis=radial_basis_spherical,
            cbf=cbf,
            scale_basis=scale_basis,
        )

    def get_bases_v(
            self,
            main_graph,
            a2a_graph,
            a2ee2a_graph,
            qint_graph,
            trip_idx_e2e,
            trip_idx_a2e,
            trip_idx_e2a,
            quad_idx,
            num_atoms,
    ):  # 修改的get_base函数， 增加使用vector向量嵌入成边特征
        """Calculate and transform basis functions."""
        m_dis = main_graph['distance']
        m_vec = main_graph['vector']
        basis_rad_main_raw = self.radial_basis(m_dis, m_vec)
        # -------
        # Calculate triplet angles
        cosφ_cab = inner_product_clamped(
            m_vec[trip_idx_e2e["out"]],
            m_vec[trip_idx_e2e["in"]],
        )
        basis_rad_cir_e2e_raw, basis_cir_e2e_raw = self.cbf_basis_tint(
            m_dis, m_vec, cosφ_cab
        )

        if self.quad_interaction:
            # Calculate quadruplet angles
            q_dis = qint_graph["distance"]
            q_vec = qint_graph["vector"]
            cosφ_cab_q, cosφ_abd, angle_cabd = self.calculate_quad_angles(
                m_vec,
                q_vec,
                quad_idx,
            )

            basis_rad_cir_qint_raw, basis_cir_qint_raw = self.cbf_basis_qint(
                q_dis, q_vec, cosφ_abd
            )
            basis_rad_sph_qint_raw, basis_sph_qint_raw = self.sbf_basis_qint(
                m_dis,
                m_vec,
                cosφ_cab_q[quad_idx["trip_out_to_quad"]],
                angle_cabd,
            )
        if a2ee2a_graph is not None:
            a2ee2a_dis = a2ee2a_graph['distance']
            a2ee2a_vec = a2ee2a_graph['vector']
        if self.atom_edge_interaction:
            basis_rad_a2ee2a_raw = self.radial_basis_aeaint(
                a2ee2a_dis,
                a2ee2a_vec
            )
            cosφ_cab_a2e = inner_product_clamped(
                m_vec[trip_idx_a2e["out"]],
                a2ee2a_vec[trip_idx_a2e["in"]],
            )
            basis_rad_cir_a2e_raw, basis_cir_a2e_raw = self.cbf_basis_aeint(
                m_dis,m_vec, cosφ_cab_a2e
            )
        if self.edge_atom_interaction:
            cosφ_cab_e2a = inner_product_clamped(
                a2ee2a_vec[trip_idx_e2a["out"]],
                m_vec[trip_idx_e2a["in"]],
            )
            basis_rad_cir_e2a_raw, basis_cir_e2a_raw = self.cbf_basis_eaint(
                a2ee2a_dis, a2ee2a_vec, cosφ_cab_e2a
            )
        if self.atom_interaction:
            a2a_dis = a2a_graph['distance']
            a2a_vec = a2a_graph['vector']
            basis_rad_a2a_raw = self.radial_basis_aint(a2a_dis, a2a_vec)
        # Shared Down Projections
        bases_qint = {}
        if self.quad_interaction:
            bases_qint["rad"] = self.mlp_rbf_qint(basis_rad_main_raw)
            bases_qint["cir"] = self.mlp_cbf_qint(
                rad_basis=basis_rad_cir_qint_raw,
                sph_basis=basis_cir_qint_raw,
                idx_sph_outer=quad_idx["triplet_in"]["out"],
            )
            bases_qint["sph"] = self.mlp_sbf_qint(
                rad_basis=basis_rad_sph_qint_raw,
                sph_basis=basis_sph_qint_raw,
                idx_sph_outer=quad_idx["out"],
                idx_sph_inner=quad_idx["out_agg"],
            )

        bases_a2e = {}
        if self.atom_edge_interaction:
            bases_a2e["rad"] = self.mlp_rbf_aeint(basis_rad_a2ee2a_raw)
            bases_a2e["cir"] = self.mlp_cbf_aeint(
                rad_basis=basis_rad_cir_a2e_raw,
                sph_basis=basis_cir_a2e_raw,
                idx_sph_outer=trip_idx_a2e["out"],
                idx_sph_inner=trip_idx_a2e["out_agg"],
            )
        bases_e2a = {}
        if self.edge_atom_interaction:
            bases_e2a["rad"] = self.mlp_rbf_eaint(basis_rad_main_raw)
            bases_e2a["cir"] = self.mlp_cbf_eaint(
                rad_basis=basis_rad_cir_e2a_raw,
                sph_basis=basis_cir_e2a_raw,
                idx_rad_outer=a2ee2a_graph["edge_index"][1],
                idx_rad_inner=a2ee2a_graph["target_neighbor_idx"],
                idx_sph_outer=trip_idx_e2a["out"],
                idx_sph_inner=trip_idx_e2a["out_agg"],
                num_atoms=num_atoms,
            )
        if self.atom_interaction:
            basis_a2a_rad = self.mlp_rbf_aint(
                rad_basis=basis_rad_a2a_raw,
                idx_rad_outer=a2a_graph["edge_index"][1],
                idx_rad_inner=a2a_graph["target_neighbor_idx"],
                num_atoms=num_atoms,
            )
        else:
            basis_a2a_rad = None

        bases_e2e = {}
        bases_e2e["rad"] = self.mlp_rbf_tint(basis_rad_main_raw)
        bases_e2e["cir"] = self.mlp_cbf_tint(
            rad_basis=basis_rad_cir_e2e_raw,
            sph_basis=basis_cir_e2e_raw,
            idx_sph_outer=trip_idx_e2e["out"],
            idx_sph_inner=trip_idx_e2e["out_agg"],
        )

        basis_atom_update = self.mlp_rbf_h(basis_rad_main_raw)
        basis_output = self.mlp_rbf_out(basis_rad_main_raw)

        return (
            basis_rad_main_raw,
            basis_atom_update,
            basis_output,
            bases_qint,
            bases_e2e,
            bases_a2e,
            bases_e2a,
            basis_a2a_rad,
        )
    def get_graphs_and_indices_lmdb(self, data):
        # main_graph使用lmdb中定义的图结构
        """ "Generate embedding and interaction graphs and indices."""
        num_atoms = data.atomic_numbers.size(0)

        # Atom interaction graph is always the largest

        (
            edge_index,
            edge_dist,
            distance_vec,
            cell_offsets,
            _,  # cell offset distances
            num_neighbors,
        ) = self.generate_graph(data, otf_graph=False)

        edge_vector = -distance_vec / edge_dist[:, None] # 对main_graph的Vector进行归一化
        # edge_vector = -distance_vec # 这里不对main_graph的Vetor进行归一化
        cell_offsets = -cell_offsets  # a - c + offset
        main_graph = {
            "edge_index": edge_index,
            "distance": edge_dist,
            "vector": edge_vector,
            "cell_offset": cell_offsets,
            "num_neighbors": num_neighbors,
        }
        id_swap = data.id_swap

        if (
                self.atom_edge_interaction
                or self.edge_atom_interaction
                or self.atom_interaction
        ):
            a2a_graph = self.generate_graph_dict(
                data, self.cutoff_aint, self.max_neighbors_aint
            )

            a2ee2a_graph = self.subselect_graph(
                data,
                a2a_graph,
                self.cutoff_aeaint,
                self.max_neighbors_aeaint,
                self.cutoff_aint,
                self.max_neighbors_aint,
            )
        else:
            main_graph = self.generate_graph_dict(
                data, self.cutoff, self.max_neighbors
            )
            a2a_graph = {}
            a2ee2a_graph = {}
        if self.quad_interaction:
            if (
                    self.atom_edge_interaction
                    or self.edge_atom_interaction
                    or self.atom_interaction
            ):
                qint_graph = self.subselect_graph_cat(
                    data,
                    a2a_graph,
                    self.cutoff_qint,
                    self.max_neighbors_qint,
                    self.cutoff_aint,
                    self.max_neighbors_aint,
                )
            else:
                assert self.cutoff_qint <= self.cutoff
                assert self.max_neighbors_qint <= self.max_neighbors
                qint_graph = self.subselect_graph_cat(
                    data,
                    main_graph,
                    self.cutoff_qint,
                    self.max_neighbors_qint,
                    self.cutoff,
                    self.max_neighbors,
                )


            # Only use quadruplets for certain tags
            self.qint_tags = self.qint_tags.to(qint_graph["edge_index"].device)
            tags_s = data.tags[qint_graph["edge_index"][0]]
            tags_t = data.tags[qint_graph["edge_index"][1]]
            qint_tag_mask_s = (tags_s[..., None] == self.qint_tags).any(dim=-1)
            qint_tag_mask_t = (tags_t[..., None] == self.qint_tags).any(dim=-1)
            qint_tag_mask = qint_tag_mask_s | qint_tag_mask_t
            qint_graph["edge_index"] = qint_graph["edge_index"][
                                       :, qint_tag_mask
                                       ]
            qint_graph["cell_offset"] = qint_graph["cell_offset"][
                                        qint_tag_mask, :
                                        ]
            qint_graph["distance"] = qint_graph["distance"][qint_tag_mask]
            qint_graph["vector"] = qint_graph["vector"][qint_tag_mask, :]
            del qint_graph["num_neighbors"]
        else:
            qint_graph = {}

        # Symmetrize edges for swapping in symmetric message passing
        # main_graph, id_swap = self.symmetrize_edges(main_graph, data.batch)
        trip_idx_e2e = get_triplets(main_graph, num_atoms=num_atoms)

        # Additional indices for quadruplets
        if self.quad_interaction:
            quad_idx = get_quadruplets(
                main_graph,
                qint_graph,
                num_atoms,
            )
        else:
            quad_idx = {}

        if self.atom_edge_interaction:
            trip_idx_a2e = get_mixed_triplets(
                a2ee2a_graph,
                main_graph,
                num_atoms=num_atoms,
                return_agg_idx=True,
            )
        else:
            trip_idx_a2e = {}
        if self.edge_atom_interaction:
            trip_idx_e2a = get_mixed_triplets(
                main_graph,
                a2ee2a_graph,
                num_atoms=num_atoms,
                return_agg_idx=True,
            )
            # a2ee2a_graph['edge_index'][1] has to be sorted for this
            a2ee2a_graph["target_neighbor_idx"] = get_inner_idx(
                a2ee2a_graph["edge_index"][1], dim_size=num_atoms
            )
        else:
            trip_idx_e2a = {}
        if self.atom_interaction:
            # a2a_graph['edge_index'][1] has to be sorted for this
            a2a_graph["target_neighbor_idx"] = get_inner_idx(
                a2a_graph["edge_index"][1], dim_size=num_atoms
            )

        return (
            main_graph,
            a2a_graph,
            a2ee2a_graph,
            qint_graph,
            id_swap,
            trip_idx_e2e,
            trip_idx_a2e,
            trip_idx_e2a,
            quad_idx,
        )

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        batch = data.batch
        atomic_numbers = data.atomic_numbers.long()
        num_atoms = atomic_numbers.shape[0]
        pos = data.pos
        if self.regress_forces and not self.direct_forces:
            pos.requires_grad_(True)
        (
            main_graph,
            a2a_graph,
            a2ee2a_graph,
            qint_graph,
            id_swap,
            trip_idx_e2e,
            trip_idx_a2e,
            trip_idx_e2a,
            quad_idx,
        ) = self.get_graphs_and_indices(data)
        self.main_graph = main_graph
        _, idx_t = main_graph["edge_index"]

        (
            basis_rad_raw,
            basis_atom_update,
            basis_output,
            bases_qint,
            bases_e2e,
            bases_a2e,
            bases_e2a,
            basis_a2a_rad,
        ) = self.get_bases_v(
            main_graph=main_graph,
            a2a_graph=a2a_graph,
            a2ee2a_graph=a2ee2a_graph,
            qint_graph=qint_graph,
            trip_idx_e2e=trip_idx_e2e,
            trip_idx_a2e=trip_idx_a2e,
            trip_idx_e2a=trip_idx_e2a,
            quad_idx=quad_idx,
            num_atoms=num_atoms,
        )
        # Embedding block
        h = self.atom_emb(atomic_numbers)
        # (nAtoms, emb_size_atom)
        m = self.edge_emb(h, basis_rad_raw, main_graph["edge_index"])

        # (nEdges, emb_size_edge)
        x_E, x_V = self.out_blocks[0](h, m, basis_output, idx_t)
        # (nEdges, emb_size_edge)
        xs_E = [x_E]
        xs_V = [x_V]

        for i in range(self.num_blocks):
            # Interaction block
            h, m = self.int_blocks[i](
                h=h,
                m=m,
                bases_qint=bases_qint,
                bases_e2e=bases_e2e,
                bases_a2e=bases_a2e,
                bases_e2a=bases_e2a,
                basis_a2a_rad=basis_a2a_rad,
                basis_atom_update=basis_atom_update,
                edge_index_main=main_graph["edge_index"],
                a2ee2a_graph=a2ee2a_graph,
                a2a_graph=a2a_graph,
                id_swap=id_swap,
                trip_idx_e2e=trip_idx_e2e,
                trip_idx_a2e=trip_idx_a2e,
                trip_idx_e2a=trip_idx_e2a,
                quad_idx=quad_idx,
            )  # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)

            x_E, x_V = self.out_blocks[i + 1](h, m, basis_output, idx_t)
            xs_E.append(x_E)
            xs_V.append(x_V)

        if self.update_v:
            x_V = self.out_mlp_V(torch.cat(xs_V, dim=-1))
            V_t = self.out_v(x_V)
            V_t = V_t.squeeze(1) # batch 3
        else:
            V_t = None
        # 坐标
        x_P = self.out_mlp_P(torch.cat(xs_E, dim=-1))
        P_t = self.out_p(x_P)
        P_t = P_t.squeeze(1)

        return V_t, P_t, main_graph

