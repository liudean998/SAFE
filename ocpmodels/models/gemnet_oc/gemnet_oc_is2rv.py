"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph
from torch_scatter import scatter, segment_coo

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    compute_neighbors,
    conditional_grad,
    get_max_neighbors_mask,
    get_pbc_distances,
    radius_graph_pbc,
    scatter_det,
)

from ocpmodels.models.base import BaseModel
from ocpmodels.models.gemnet_oc.gemnet_oc import GemNetOC
from ocpmodels.modules.scaling.compat import load_scales_compat
from .initializers import get_initializer
from .interaction_indices import (
    get_mixed_triplets,
    get_quadruplets,
    get_triplets,
)
from .layers.atom_update_block import OutputBlockMask
from .layers.base_layers import Dense, ResidualLayer
from .layers.efficient import BasisEmbedding
from .layers.embedding_block import AtomEmbedding, EdgeEmbedding, AtomEmbeddingTags, EdgeEmbeddingVectorDis
from .layers.force_scaler import ForceScaler
from .layers.interaction_block import InteractionBlock
from .layers.radial_basis import RadialBasis
from .layers.spherical_basis import CircularBasisLayer, SphericalBasisLayer
from .utils import (
    get_angle,
    get_edge_id,
    get_inner_idx,
    inner_product_clamped,
    mask_neighbors,
    repeat_blocks,
)


@registry.register_model("gemnet_is2rv")
class GemNetOCRV(GemNetOC):
    """
    Arguments
    ---------
    num_atoms (int): Unused argument
    bond_feat_dim (int): Unused argument
    num_targets: int
        Number of prediction targets.

    num_spherical: int
        Controls maximum frequency.
    num_radial: int
        Controls maximum frequency.
    num_blocks: int
        Number of building blocks to be stacked.

    emb_size_atom: int
        Embedding size of the atoms.
    emb_size_edge: int
        Embedding size of the edges.
    emb_size_trip_in: int
        (Down-projected) embedding size of the quadruplet edge embeddings
        before the bilinear layer.
    emb_size_trip_out: int
        (Down-projected) embedding size of the quadruplet edge embeddings
        after the bilinear layer.
    emb_size_quad_in: int
        (Down-projected) embedding size of the quadruplet edge embeddings
        before the bilinear layer.
    emb_size_quad_out: int
        (Down-projected) embedding size of the quadruplet edge embeddings
        after the bilinear layer.
    emb_size_aint_in: int
        Embedding size in the atom interaction before the bilinear layer.
    emb_size_aint_out: int
        Embedding size in the atom interaction after the bilinear layer.
    emb_size_rbf: int
        Embedding size of the radial basis transformation.
    emb_size_cbf: int
        Embedding size of the circular basis transformation (one angle).
    emb_size_sbf: int
        Embedding size of the spherical basis transformation (two angles).

    num_before_skip: int
        Number of residual blocks before the first skip connection.
    num_after_skip: int
        Number of residual blocks after the first skip connection.
    num_concat: int
        Number of residual blocks after the concatenation.
    num_atom: int
        Number of residual blocks in the atom embedding blocks.
    num_output_afteratom: int
        Number of residual blocks in the output blocks
        after adding the atom embedding.
    num_atom_emb_layers: int
        Number of residual blocks for transforming atom embeddings.
    num_global_out_layers: int
        Number of final residual blocks before the output.

    regress_forces: bool
        Whether to predict forces. Default: True
    direct_forces: bool
        If True predict forces based on aggregation of interatomic directions.
        If False predict forces based on negative gradient of energy potential.
    use_pbc: bool
        Whether to use periodic boundary conditions.
    scale_backprop_forces: bool
        Whether to scale up the energy and then scales down the forces
        to prevent NaNs and infs in backpropagated forces.

    cutoff: float
        Embedding cutoff for interatomic connections and embeddings in Angstrom.
    cutoff_qint: float
        Quadruplet interaction cutoff in Angstrom.
        Optional. Uses cutoff per default.
    cutoff_aeaint: float
        Edge-to-atom and atom-to-edge interaction cutoff in Angstrom.
        Optional. Uses cutoff per default.
    cutoff_aint: float
        Atom-to-atom interaction cutoff in Angstrom.
        Optional. Uses maximum of all other cutoffs per default.
    max_neighbors: int
        Maximum number of neighbors for interatomic connections and embeddings.
    max_neighbors_qint: int
        Maximum number of quadruplet interactions per embedding.
        Optional. Uses max_neighbors per default.
    max_neighbors_aeaint: int
        Maximum number of edge-to-atom and atom-to-edge interactions per embedding.
        Optional. Uses max_neighbors per default.
    max_neighbors_aint: int
        Maximum number of atom-to-atom interactions per atom.
        Optional. Uses maximum of all other neighbors per default.
    enforce_max_neighbors_strictly: bool
        When subselected edges based on max_neighbors args, arbitrarily
        select amongst degenerate edges to have exactly the correct number.
    rbf: dict
        Name and hyperparameters of the radial basis function.
    rbf_spherical: dict
        Name and hyperparameters of the radial basis function used as part of the
        circular and spherical bases.
        Optional. Uses rbf per default.
    envelope: dict
        Name and hyperparameters of the envelope function.
    cbf: dict
        Name and hyperparameters of the circular basis function.
    sbf: dict
        Name and hyperparameters of the spherical basis function.
    extensive: bool
        Whether the output should be extensive (proportional to the number of atoms)
    forces_coupled: bool
        If True, enforce that |F_st| = |F_ts|. No effect if direct_forces is False.
    output_init: str
        Initialization method for the final dense layer.
    activation: str
        Name of the activation function.
    scale_file: str
        Path to the pytorch file containing the scaling factors.

    quad_interaction: bool
        Whether to use quadruplet interactions (with dihedral angles)
    atom_edge_interaction: bool
        Whether to use atom-to-edge interactions
    edge_atom_interaction: bool
        Whether to use edge-to-atom interactions
    atom_interaction: bool
        Whether to use atom-to-atom interactions

    scale_basis: bool
        Whether to use a scaling layer in the raw basis function for better
        numerical stability.
    qint_tags: list
        Which atom tags to use quadruplet interactions for.
        0=sub-surface bulk, 1=surface, 2=adsorbate atoms.
    """

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
            edge_mask: bool =False,
            edge_mask_tags = 0,
            loop_num_v: int = 1, # Vetor输出迭代的次数
            loop1_step: int = None,
            loop2_step: int = None,
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
        # self.atom_emb_tags = AtomEmbeddingTags(emb_size_atom, num_elements, tags_size=64)
        # self.atom_emb = AtomEmbedding(emb_size_atom, atom_features)  # 修改的地方
        self.edge_emb = EdgeEmbedding(
            emb_size_atom, num_radial, emb_size_edge, activation=activation
        )
        # self.edge_emb = EdgeEmbeddingVectorDis(
        #     emb_size_atom, num_radial, emb_size_edge, activation=activation
        # )

        # Interaction Blocks
        int_blocks = []
        for _ in range(num_blocks*loop_num_v):
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
        for _ in range((num_blocks + 1)*loop_num_v):
            out_blocks.append(
                OutputBlockMask(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_rbf=emb_size_rbf,
                    nHidden=num_atom,
                    nHidden_afteratom=num_output_afteratom,
                    activation=activation,
                    direct_forces=direct_forces,
                    update_v=update_v, # 修改
                )
            )
        self.out_blocks = torch.nn.ModuleList(out_blocks)

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
        out_initializer = get_initializer(output_init)
        self.out_v.reset_parameters(out_initializer)
        self.edge_mask = edge_mask
        self.edge_mask_tags = edge_mask_tags
        self.loop_num_v = loop_num_v
        self.loop1_step = loop1_step
        self.loop1_freez = False
        self.loop2_step = loop2_step
        load_scales_compat(self, scale_file)

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
            num_radial=int(num_radial/4),
            cutoff=self.cutoff,
            rbf=rbf,
            envelope=envelope,
            scale_basis=scale_basis,
        ) # 修改
        radial_basis_spherical = RadialBasis(
            num_radial=int(num_radial/4),
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
                num_radial=int(num_radial/4),
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
                num_radial=int(num_radial/4),
                cutoff=self.cutoff_aeaint,
                rbf=rbf,
                envelope=envelope,
                scale_basis=scale_basis,
            )
            radial_basis_spherical_aeaint = RadialBasis(
                num_radial=int(num_radial/4),
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
                num_radial=int(num_radial/4),
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
            # a2a_graph = self.generate_graph_dict(
            #     data, self.cutoff_aint, self.max_neighbors_aint
            # )
            a2a_graph = main_graph

            # main_graph = self.subselect_graph(
            #     data,
            #     a2a_graph,
            #     self.cutoff,
            #     self.max_neighbors,
            #     self.cutoff_aint,
            #     self.max_neighbors_aint,
            # )

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

    def subselect_graph_from_main_graph_V(self, data, main_graph):
        num_atoms = data.atomic_numbers.size(0)
        a2a_graph = main_graph
        if (
                self.atom_edge_interaction
                or self.edge_atom_interaction
                or self.atom_interaction
        ):
            a2ee2a_graph = self.subselect_graph(
                data,
                a2a_graph,
                self.cutoff_aeaint,
                self.max_neighbors_aeaint*2, # 由于这里使用了main_graph作为子图提取，先增大这里，需要的话，可以将原有函数也改为使用main_graph获取子图
                self.cutoff_aint,
                self.max_neighbors_aint,
            )
        else:
            a2ee2a_graph = {}
        trip_idx_e2e = get_triplets(main_graph, num_atoms=num_atoms)
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
            trip_idx_e2e,
            trip_idx_a2e,
            trip_idx_e2a,
        )

    @staticmethod
    def reshape_vector(vector):
        size = vector.shape
        vector = vector.view(int(size[0]/4),
                             size[1]*4)
        return vector

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
    ): # 修改的get_base函数， 增加使用vector向量嵌入成边特征
        """Calculate and transform basis functions."""
        # basis_rad_main_raw = self.radial_basis(main_graph["distance"])
        # 修改-增加
        # vector = main_graph["vector"]
        vector_dis = torch.cat((main_graph['vector'], main_graph['distance'].unsqueeze(1)), dim=1)
        vector_dis = vector_dis.view(-1)
        basis_rad_main_raw_v = self.radial_basis(vector_dis)
        # print(size)
        basis_rad_main_raw_v = self.reshape_vector(basis_rad_main_raw_v)
        basis_rad_main_raw = basis_rad_main_raw_v.clone()
        # -------
        # Calculate triplet angles
        cosφ_cab = inner_product_clamped(
            main_graph["vector"][trip_idx_e2e["out"]],
            main_graph["vector"][trip_idx_e2e["in"]],
        )
        basis_rad_cir_e2e_raw, basis_cir_e2e_raw = self.cbf_basis_tint(
            vector_dis, cosφ_cab
        )
        basis_rad_cir_e2e_raw = self.reshape_vector(basis_rad_cir_e2e_raw)

        if self.quad_interaction:
            # Calculate quadruplet angles
            cosφ_cab_q, cosφ_abd, angle_cabd = self.calculate_quad_angles(
                main_graph["vector"],
                qint_graph["vector"],
                quad_idx,
            )

            basis_rad_cir_qint_raw, basis_cir_qint_raw = self.cbf_basis_qint(
                qint_graph["distance"], cosφ_abd
            )
            basis_rad_sph_qint_raw, basis_sph_qint_raw = self.sbf_basis_qint(
                main_graph["distance"],
                cosφ_cab_q[quad_idx["trip_out_to_quad"]],
                angle_cabd,
            )
        if self.atom_edge_interaction:
            a2ee2a_vector_dis = torch.cat((a2ee2a_graph['vector'],
                                           a2ee2a_graph['distance'].unsqueeze(1)),
                                          dim=1)
            a2ee2a_vector_dis = a2ee2a_vector_dis.view(-1)
            basis_rad_a2ee2a_raw = self.radial_basis_aeaint(
                a2ee2a_vector_dis
            )
            basis_rad_a2ee2a_raw = self.reshape_vector(basis_rad_a2ee2a_raw)
            cosφ_cab_a2e = inner_product_clamped(
                main_graph["vector"][trip_idx_a2e["out"]],
                a2ee2a_graph["vector"][trip_idx_a2e["in"]],
            )
            basis_rad_cir_a2e_raw, basis_cir_a2e_raw = self.cbf_basis_aeint(
                vector_dis, cosφ_cab_a2e
            )
            basis_rad_cir_a2e_raw = self.reshape_vector(basis_rad_cir_a2e_raw)
        if self.edge_atom_interaction:
            cosφ_cab_e2a = inner_product_clamped(
                a2ee2a_graph["vector"][trip_idx_e2a["out"]],
                main_graph["vector"][trip_idx_e2a["in"]],
            )
            basis_rad_cir_e2a_raw, basis_cir_e2a_raw = self.cbf_basis_eaint(
                a2ee2a_vector_dis, cosφ_cab_e2a
            )
            basis_rad_cir_e2a_raw = self.reshape_vector(basis_rad_cir_e2a_raw)
        if self.atom_interaction:
            basis_rad_a2a_raw = self.radial_basis_aint(vector_dis) # 原是a2a_graph distance, 由于这里使a2a_graph=main_graph
            basis_rad_a2a_raw = self.reshape_vector(basis_rad_a2a_raw)
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
            basis_rad_main_raw_v # 修改--增加
        )

    def change_layers_grad(self,
                           grad: bool,
                           num_blocks: int,
                           inter_blocks: list):
        all_layers = []
        out_s = self.out_blocks[int(inter_blocks[0] * (num_blocks+1) / num_blocks)]
        all_layers.append(out_s)
        for block in inter_blocks:
            all_layers.append(self.int_blocks[block])
        for layer in all_layers:
            layer.requires_grad_(grad)


    @conditional_grad(torch.enable_grad())
    def forward(self, data, step=None):
        # --------------------输出修改为pos更新量----------------------------------
        tags = data.tags
        atomic_numbers = data.atomic_numbers.long()
        num_atoms = atomic_numbers.shape[0]
        # loop_V = []

        for n_v in range(self.loop_num_v):
            if n_v == 0:
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
                ) = self.get_graphs_and_indices_lmdb(data) # 使用数据集中的图结构为main_graph
            else:
                (
                    main_graph,
                    a2a_graph,
                    a2ee2a_graph,
                    trip_idx_e2e,
                    trip_idx_a2e,
                    trip_idx_e2a,
                ) = self.subselect_graph_from_main_graph_V(data, main_graph)  # 使用mian_graph生成子图
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
                basis_rad_raw_v # 修改-增加
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
            # h = self.atom_emb_tags(atomic_numbers, tags)
            # (nAtoms, emb_size_atom)
            # m = self.edge_emb(h, basis_rad_raw_v, basis_rad_raw, main_graph["edge_index"])
            m = self.edge_emb(h, basis_rad_raw_v, main_graph["edge_index"])

            # 对边进行筛选
            if self.edge_mask:
                # 筛选去除0-0的边，因为0-0的边不会进行更新
                tags12 = torch.where(tags>self.edge_mask_tags)[0]
                src, dst = main_graph['edge_index']
                cond1 = torch.isin(src, tags12)
                cond2 = torch.isin(dst, tags12)
                # 结合所有条件，生成一个布尔掩码
                mask = cond1 | cond2
            else:
                mask = None
            # print(mask)
            # (nEdges, emb_size_edge)
            x_E, x_F, x_V = (self.out_blocks[(self.num_blocks+1)*n_v]
                             (h, m,
                              basis_output, idx_t,
                              mask=mask, out_energy=False,
                              out_vector=True))
            # (nEdges, emb_size_edge)
            xs_V = [x_V]
            # if torch.isnan(x_V).any():
            #     print(x_V,'????')

            for i in range(self.num_blocks):
                # Interaction block
                h, m = self.int_blocks[(n_v*self.num_blocks)+i](
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

                x_E, x_F, x_V = self.out_blocks[(n_v*self.num_blocks)+i + 1](h, m, basis_output, idx_t, mask=mask,
                                                       out_energy=False,
                                                       out_vector=True)
                if torch.isnan(x_V).any():
                    print(x_V, 'i')
                # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)
                xs_V.append(x_V)
            x_V = self.out_mlp_V(torch.cat(xs_V, dim=-1))
            V_t = self.out_v(x_V)
            V = V_t.squeeze(1) # batch 3
            # delta1
            # delta_v = (n_v + 1)*(V_t - data.v0)/self.loop_num_v
            # delta2
            # delta_v = (n_v+1*V_t)/self.loop_num_v
            # if mask is not None:
            #     V = main_graph['vector'][mask] + delta_v
            #     main_graph['vector'][mask] = V
            # else:
            #     V = main_graph['vector'] + delta_v
            #     main_graph['vector'] = V
            # 直接输出Vetor
            # V = V_t
            if mask is not None:
                main_graph['vector'][mask] = V
            else:
                main_graph['vector'] = V
            # print(n_v)
            # loop_V.append(V)
            if step and self.loop1_step > step and n_v == 0:
                break
            elif step:
                if self.loop1_freez is False:
                    self.change_layers_grad(False,
                                            3,
                                            [0, 1, 2])
            # if step and self.loop2_step > step and n_v == 1:
            #     # print('1_break')
            #     break
            # # if torch.isnan(V_t).any():
            # #     print(n_v)
            # #     print(V_t)
            # #     import time
            # #     time.sleep(10000)

        return V, mask

@registry.register_model("gemnet_is2rve")
class GemNetOCRVE(GemNetOC):
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
        self.atom_emb = AtomEmbedding(emb_size_atom, 83)
        self.atom_emb_tags = AtomEmbeddingTags(emb_size_atom, 83, tags_size=64)
        # self.atom_emb = AtomEmbedding(emb_size_atom, atom_features)  # 修改的地方
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
                OutputBlockMask(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_rbf=emb_size_rbf,
                    nHidden=num_atom,
                    nHidden_afteratom=num_output_afteratom,
                    activation=activation,
                    direct_forces=direct_forces,
                    update_v=update_v, # 修改
                )
            )
        self.out_blocks = torch.nn.ModuleList(out_blocks)

        out_mlp_E = [
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
        self.out_mlp_E = torch.nn.Sequential(*out_mlp_E)
        self.out_energy = Dense(
            emb_size_atom, num_targets, bias=False, activation=None
        )
        if direct_forces:
            out_mlp_F = [
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
            self.out_mlp_F = torch.nn.Sequential(*out_mlp_F)
            self.out_forces = Dense(
                emb_size_edge, num_targets, bias=False, activation=None
            )
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
        out_initializer = get_initializer(output_init)
        self.out_energy.reset_parameters(out_initializer)
        if direct_forces:
            self.out_forces.reset_parameters(out_initializer)
        if update_v:
            self.out_v.reset_parameters(out_initializer)

        load_scales_compat(self, scale_file)


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
        self.radial_basis = RadialBasis(
            num_radial=num_radial,
            cutoff=self.cutoff,
            rbf=rbf,
            envelope=envelope,
            scale_basis=scale_basis,
        )
        self.radial_basis_main = RadialBasis(
            num_radial=int(num_radial/3),
            cutoff=self.cutoff,
            rbf=rbf,
            envelope=envelope,
            scale_basis=scale_basis,
        ) # 修改
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

            # main_graph = self.subselect_graph(
            #     data,
            #     a2a_graph,
            #     self.cutoff,
            #     self.max_neighbors,
            #     self.cutoff_aint,
            #     self.max_neighbors_aint,
            # )

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
    ): # 修改的get_base函数， 增加使用vector向量嵌入成边特征
        """Calculate and transform basis functions."""
        basis_rad_main_raw = self.radial_basis(main_graph["distance"])
        # 修改-增加
        vector = main_graph["vector"]
        vector = vector.view(-1)
        basis_rad_main_raw_v = self.radial_basis_main(vector)
        size = basis_rad_main_raw_v.shape
        # print(size)
        basis_rad_main_raw_v = basis_rad_main_raw_v.view(int(size[0]/3),
                                                         size[1]*3)
        # -------
        # Calculate triplet angles
        cosφ_cab = inner_product_clamped(
            main_graph["vector"][trip_idx_e2e["out"]],
            main_graph["vector"][trip_idx_e2e["in"]],
        )
        basis_rad_cir_e2e_raw, basis_cir_e2e_raw = self.cbf_basis_tint(
            main_graph["distance"], cosφ_cab
        )

        if self.quad_interaction:
            # Calculate quadruplet angles
            cosφ_cab_q, cosφ_abd, angle_cabd = self.calculate_quad_angles(
                main_graph["vector"],
                qint_graph["vector"],
                quad_idx,
            )

            basis_rad_cir_qint_raw, basis_cir_qint_raw = self.cbf_basis_qint(
                qint_graph["distance"], cosφ_abd
            )
            basis_rad_sph_qint_raw, basis_sph_qint_raw = self.sbf_basis_qint(
                main_graph["distance"],
                cosφ_cab_q[quad_idx["trip_out_to_quad"]],
                angle_cabd,
            )
        if self.atom_edge_interaction:
            basis_rad_a2ee2a_raw = self.radial_basis_aeaint(
                a2ee2a_graph["distance"]
            )
            cosφ_cab_a2e = inner_product_clamped(
                main_graph["vector"][trip_idx_a2e["out"]],
                a2ee2a_graph["vector"][trip_idx_a2e["in"]],
            )
            basis_rad_cir_a2e_raw, basis_cir_a2e_raw = self.cbf_basis_aeint(
                main_graph["distance"], cosφ_cab_a2e
            )
        if self.edge_atom_interaction:
            cosφ_cab_e2a = inner_product_clamped(
                a2ee2a_graph["vector"][trip_idx_e2a["out"]],
                main_graph["vector"][trip_idx_e2a["in"]],
            )
            basis_rad_cir_e2a_raw, basis_cir_e2a_raw = self.cbf_basis_eaint(
                a2ee2a_graph["distance"], cosφ_cab_e2a
            )
        if self.atom_interaction:
            basis_rad_a2a_raw = self.radial_basis_aint(a2a_graph["distance"])

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
            basis_rad_main_raw_v # 修改--增加
        )

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        # --------------------输出修改为pos更新量----------------------------------
        pos = data.pos
        batch = data.batch
        tags = data.tags
        atomic_numbers = data.atomic_numbers.long()
        num_atoms = atomic_numbers.shape[0]

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
        ) = self.get_graphs_and_indices_lmdb(data) # 使用数据集中的图结构为main_graph
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
            basis_rad_raw_v # 修改-增加
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
        h = self.atom_emb_tags(atomic_numbers, tags)
        # (nAtoms, emb_size_atom)
        m = self.edge_emb(h, basis_rad_raw_v, main_graph["edge_index"])
        # 筛选去除0-0的边，因为0-0的边不会进行更新
        # tags12 = torch.where(tags>0)[0]
        # src, dst = main_graph['edge_index']
        # cond1 = torch.isin(src, tags12)
        # cond2 = torch.isin(dst, tags12)
        # # 结合所有条件，生成一个布尔掩码
        # mask = cond1 & cond2
        #
        # tags_mask = tags > 0
        tags_mask = None
        mask = None
        # (nEdges, emb_size_edge)
        x_E, x_F, x_V = self.out_blocks[0](h, m, basis_output, idx_t,
                                           mask=mask, tags_mask=tags_mask,
                                           out_vector=True)
        # (nEdges, emb_size_edge)
        xs_V = [x_V]
        xs_E = [x_E]

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

            x_E, x_F, x_V = self.out_blocks[i + 1](h, m, basis_output, idx_t,
                                                   mask=mask, tags_mask=tags_mask,
                                                   out_vector=True)
            # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)
            xs_V.append(x_V)
            xs_E.append(x_E)


        # 能量预测部分
        x_E = self.out_mlp_E(torch.cat(xs_E, dim=-1))
        with torch.cuda.amp.autocast(False):
            E_t = self.out_energy(x_E.float())
        nMolecules = torch.max(batch) + 1
        if tags_mask is not None:
            E_batch = batch[tags_mask]
        else:
            E_batch = batch
        if self.extensive:
            E_t = scatter_det(
                E_t, E_batch, dim=0, dim_size=nMolecules, reduce="add"
            )  # (nMolecules, num_targets)
        else:
            E_t = scatter_det(
                E_t, E_batch, dim=0, dim_size=nMolecules, reduce="mean"
            )  # (nMolecules, num_targets)
        E_t = E_t.squeeze(1)  # (num_molecules)
        # Global output block for final predictions
        if self.update_v:
            x_V = self.out_mlp_V(torch.cat(xs_V, dim=-1))
            V_t = self.out_v(x_V)
            # P_t = P_t.squeeze(1) # batch 3
            V_t = V_t.squeeze(1) # batch 3
        else:
            V_t = 0

        return V_t, mask, E_t

@registry.register_model("gemnet_is2rv2e")
class GemNetOCRV2E(GemNetOC):
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
            # qint_tags: list = [0, 1, 2],
            qint_tags: list = None,
            num_elements: int = 83,
            otf_graph: bool = False,
            scale_file: Optional[str] = None,
            update_v: bool = True, # 是否对main_graph vector进行训练
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
        # self.atom_emb = AtomEmbedding(emb_size_atom, 83)
        self.atom_emb_tags = AtomEmbeddingTags(emb_size_atom, num_elements, tags_size=64)
        # self.atom_emb = AtomEmbedding(emb_size_atom, atom_features)  # 修改的地方
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
        for _ in range(num_blocks + 2):
            out_blocks.append(
                OutputBlockMask(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_rbf=emb_size_rbf,
                    nHidden=num_atom,
                    nHidden_afteratom=num_output_afteratom,
                    activation=activation,
                    direct_forces=direct_forces,
                    update_v=update_v, # 修改
                )
            )
        self.out_blocks = torch.nn.ModuleList(out_blocks)

        # 能量输出
        out_mlp_E = [
                        Dense(
                            # emb_size_atom * (num_blocks + 1),
                            emb_size_atom * (3 + 1), # 先固定层数
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
        self.out_mlp_E = torch.nn.Sequential(*out_mlp_E)
        self.out_energy = Dense(
            emb_size_atom, num_targets, bias=False, activation=None
        )
        if direct_forces:
            out_mlp_F = [
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
            self.out_mlp_F = torch.nn.Sequential(*out_mlp_F)
            self.out_forces = Dense(
                emb_size_edge, num_targets, bias=False, activation=None
            )
        if self.update_v:
            # vecor输出
            out_mlp_V = [
                            Dense(
                                # emb_size_edge * (num_blocks + 1),
                                # 先固定层数，Vetor交互两次，输出三次，energy交互三次输出4次
                                emb_size_edge * (2 + 1),
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
        out_initializer = get_initializer(output_init)
        self.out_energy.reset_parameters(out_initializer)
        if direct_forces:
            self.out_forces.reset_parameters(out_initializer)
        if update_v:
            self.out_v.reset_parameters(out_initializer)

        load_scales_compat(self, scale_file)

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
        self.radial_basis = RadialBasis(
            num_radial=num_radial,
            cutoff=self.cutoff,
            rbf=rbf,
            envelope=envelope,
            scale_basis=scale_basis,
        )
        self.radial_basis_main = RadialBasis(
            num_radial=int(num_radial/3),
            cutoff=self.cutoff,
            rbf=rbf,
            envelope=envelope,
            scale_basis=scale_basis,
        ) # 修改
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

            # main_graph = self.subselect_graph(
            #     data,
            #     a2a_graph,
            #     self.cutoff,
            #     self.max_neighbors,
            #     self.cutoff_aint,
            #     self.max_neighbors_aint,
            # )

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
                qint_graph = self.subselect_graph(
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
                qint_graph = self.subselect_graph(
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

    def sort_by_edge_index(self, graph):
        edge_index = graph["edge_index"]
        distance = graph["distance"]
        vector = graph["vector"]
        cell_offset = graph["cell_offset"]

        # 获取 edge_index[1] 的排序索引
        sort_idx = edge_index[1].argsort()

        # 根据排序索引重新排序张量
        sorted_edge_index = edge_index[:, sort_idx]
        sorted_distance = distance[sort_idx]
        sorted_vector = vector[sort_idx]
        sorted_cell_offset = cell_offset[sort_idx]
        # 返回排序后的字典
        return {
            'edge_index': sorted_edge_index,
            'distance': sorted_distance,
            'vector': sorted_vector,
            'cell_offset': sorted_cell_offset,
            'num_neighbors': graph['num_neighbors'],  # num_neighbors 不变
        }
    def subselect_graph_from_main_graph(self, data, main_graph):
        num_atoms = data.atomic_numbers.size(0)
        a2a_graph = self.subselect_graph(
            data,
            main_graph,
            self.cutoff_aint-2, # 6
            self.max_neighbors_aint, # 1000
            self.cutoff,
            self.max_neighbors_aint
        )
        main_graph = self.subselect_graph(
            data,
            main_graph,
            self.cutoff_aint,
            self.max_neighbors,
            self.cutoff,
            self.max_neighbors_aint
        )
        if (
                self.atom_edge_interaction
                or self.edge_atom_interaction
                or self.atom_interaction
        ):
            a2ee2a_graph = self.subselect_graph(
                data,
                a2a_graph,
                self.cutoff_aeaint,
                self.max_neighbors_aeaint,
                self.cutoff_aint,
                self.max_neighbors_aint,
            )
        else:
            a2ee2a_graph = {}
        main_graph, id_swap = self.symmetrize_edges(main_graph, data.batch)
        trip_idx_e2e = get_triplets(main_graph, num_atoms=num_atoms)
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
            id_swap,
            trip_idx_e2e,
            trip_idx_a2e,
            trip_idx_e2a,
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
    ): # 修改的get_base函数， 增加使用vector向量嵌入成边特征
        """Calculate and transform basis functions."""
        basis_rad_main_raw = self.radial_basis(main_graph["distance"])
        # 修改-增加
        vector = main_graph["vector"]
        vector = vector.view(-1)
        basis_rad_main_raw_v = self.radial_basis_main(vector)
        size = basis_rad_main_raw_v.shape
        # print(size)
        basis_rad_main_raw_v = basis_rad_main_raw_v.view(int(size[0]/3),
                                                         size[1]*3)
        # -------
        # Calculate triplet angles
        cosφ_cab = inner_product_clamped(
            main_graph["vector"][trip_idx_e2e["out"]],
            main_graph["vector"][trip_idx_e2e["in"]],
        )
        basis_rad_cir_e2e_raw, basis_cir_e2e_raw = self.cbf_basis_tint(
            main_graph["distance"], cosφ_cab
        )

        if self.quad_interaction:
            # Calculate quadruplet angles
            cosφ_cab_q, cosφ_abd, angle_cabd = self.calculate_quad_angles(
                main_graph["vector"],
                qint_graph["vector"],
                quad_idx,
            )

            basis_rad_cir_qint_raw, basis_cir_qint_raw = self.cbf_basis_qint(
                qint_graph["distance"], cosφ_abd
            )
            basis_rad_sph_qint_raw, basis_sph_qint_raw = self.sbf_basis_qint(
                main_graph["distance"],
                cosφ_cab_q[quad_idx["trip_out_to_quad"]],
                angle_cabd,
            )
        if self.atom_edge_interaction:
            basis_rad_a2ee2a_raw = self.radial_basis_aeaint(
                a2ee2a_graph["distance"]
            )
            cosφ_cab_a2e = inner_product_clamped(
                main_graph["vector"][trip_idx_a2e["out"]],
                a2ee2a_graph["vector"][trip_idx_a2e["in"]],
            )
            basis_rad_cir_a2e_raw, basis_cir_a2e_raw = self.cbf_basis_aeint(
                main_graph["distance"], cosφ_cab_a2e
            )
        if self.edge_atom_interaction:
            cosφ_cab_e2a = inner_product_clamped(
                a2ee2a_graph["vector"][trip_idx_e2a["out"]],
                main_graph["vector"][trip_idx_e2a["in"]],
            )
            basis_rad_cir_e2a_raw, basis_cir_e2a_raw = self.cbf_basis_eaint(
                a2ee2a_graph["distance"], cosφ_cab_e2a
            )
        if self.atom_interaction:
            basis_rad_a2a_raw = self.radial_basis_aint(a2a_graph["distance"])

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
            basis_rad_main_raw_v # 修改--增加
        )

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        # --------------------energy使用更新后的V_t来预测----------------------------------
        pos = data.pos
        batch = data.batch
        tags = data.tags
        atomic_numbers = data.atomic_numbers.long()
        num_atoms = atomic_numbers.shape[0]
        # ----------------- 更新Vetor -------------------------
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
        ) = self.get_graphs_and_indices_lmdb(data) # 使用数据集中的图结构为main_graph
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
            basis_rad_raw_v # 修改-增加
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
        h = self.atom_emb_tags(atomic_numbers, tags)
        # (nAtoms, emb_size_atom)
        m = self.edge_emb(h, basis_rad_raw_v, main_graph["edge_index"])
        # 筛选去除0-0的边，因为0-0的边不会进行更新

        x_E, x_F, x_V = self.out_blocks[0](h, m, basis_output, idx_t,
                                           out_vector=True,
                                           out_energy=False)
        # (nEdges, emb_size_edge)
        xs_V = [x_V]
        # xs_E = [x_E] # 先以原始数据输出一次E

        for i in range(2): # 先固定层数
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

            x_E, x_F, x_V = self.out_blocks[i + 1](h, m, basis_output, idx_t,
                                                   out_energy=False,
                                                   out_vector=True)
            # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)
            xs_V.append(x_V)
        if self.update_v:
            x_V = self.out_mlp_V(torch.cat(xs_V, dim=-1))
            V_t = self.out_v(x_V)
            # P_t = P_t.squeeze(1) # batch 3
            V_t = V_t.squeeze(1) # batch 3
        else:
            V_t = 0

        # ----------------- 能量预测部分 --------------------------------
        main_graph['vector'] = V_t
        main_graph =self.sort_by_edge_index(main_graph)
        (
            main_graph_e,
            a2a_graph_e,
            a2ee2a_graph_e,
            id_swap,
            trip_idx_e2e_e,
            trip_idx_a2e_e,
            trip_idx_e2a_e,
        ) = self.subselect_graph_from_main_graph(data, main_graph)
        _, idx_t_e = main_graph_e["edge_index"]

        (
            basis_rad_raw_e,
            basis_atom_update_e,
            basis_output_e,
            _,
            bases_e2e_e,
            bases_a2e_e,
            bases_e2a_e,
            basis_a2a_rad_e,
        ) = self.get_bases(
            main_graph=main_graph_e,
            a2a_graph=a2a_graph_e,
            a2ee2a_graph=a2ee2a_graph_e,
            qint_graph={},
            trip_idx_e2e=trip_idx_e2e_e,
            trip_idx_a2e=trip_idx_a2e_e,
            trip_idx_e2a=trip_idx_e2a_e,
            quad_idx=None,
            num_atoms=num_atoms,
        )
        h_e = self.atom_emb(atomic_numbers)
        # print(basis_rad_raw_e.shape, main_graph['distance'].shape, main_graph['edge_index'].shape)
        m_e = self.edge_emb(h_e, basis_rad_raw_e, main_graph_e['edge_index'])
        xs_E = []
        x_E, x_F = self.out_blocks[3](h_e, m_e, basis_output_e, idx_t_e,
                                           out_vector=False)
        xs_E.append(x_E)
        for i in range(2, 5):
            h_e, m_e = self.int_blocks[i](
                h=h_e,
                m=m_e,
                bases_qint=None,
                bases_e2e=bases_e2e_e,
                bases_a2e=bases_a2e_e,
                bases_e2a=bases_e2a_e,
                basis_a2a_rad=basis_a2a_rad_e,
                basis_atom_update=basis_atom_update_e,
                edge_index_main=main_graph_e["edge_index"],
                a2ee2a_graph=a2ee2a_graph_e,
                a2a_graph=a2a_graph_e,
                id_swap=id_swap,
                trip_idx_e2e=trip_idx_e2e_e,
                trip_idx_a2e=trip_idx_a2e_e,
                trip_idx_e2a=trip_idx_e2a_e,
                quad_idx=None,
            )
            x_E, x_F = self.out_blocks[i+1](h_e, m_e, basis_output_e, idx_t_e)
            xs_E.append(x_E)
        # for i in xs_E:
        #     print(i.shape)
        # print(torch.cat(xs_E, dim=-1).shape)
        x_E = self.out_mlp_E(torch.cat(xs_E, dim=-1))
        with torch.cuda.amp.autocast(False):
            E_t = self.out_energy(x_E.float())
        nMolecules = torch.max(batch) + 1
        if self.extensive:
            E_t = scatter_det(
                E_t, batch, dim=0, dim_size=nMolecules, reduce="add"
            )  # (nMolecules, num_targets)
        else:
            E_t = scatter_det(
                E_t, batch, dim=0, dim_size=nMolecules, reduce="mean"
            )  # (nMolecules, num_targets)
        E_t = E_t.squeeze(1)  # (num_molecules)
        # Global output block for final predictions
        mask = None
        return V_t, mask, E_t
