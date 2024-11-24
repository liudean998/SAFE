"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .base_layers import Dense
from .static.atom_info import atom_infos as atom_info_normal


class AtomEmbedding(torch.nn.Module):
    """
    Initial atom embeddings based on the atom type

    Arguments
    ---------
    emb_size: int
        Atom embeddings size
    """
# 原
    def __init__(self, emb_size: int, num_elements: int) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.embeddings = torch.nn.Embedding(num_elements, emb_size)
        # init by uniform distribution
        torch.nn.init.uniform_(
            self.embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3)
        )

    def forward(self, Z) -> torch.Tensor:
        """
        Returns
        -------
        h: torch.Tensor, shape=(nAtoms, emb_size)
            Atom embeddings.
        """
        h = self.embeddings(Z - 1)  # -1 because Z.min()=1 (==Hydrogen)
        return h

# # 增加原子标签
#     def __init__(self, emb_size: int, num_elements: int) -> None:
#         super().__init__()
#         self.emb_size = emb_size
#         self.embeddings = torch.nn.Embedding(num_elements, emb_size-128)
#         # init by uniform distribution
#         torch.nn.init.uniform_(
#             self.embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3)
#         )
#         self.embeddings_tags = torch.nn.Embedding(3, 128)
#         # init by uniform distribution
#         torch.nn.init.uniform_(
#             self.embeddings_tags.weight, a=-np.sqrt(3), b=np.sqrt(3)
#         )
#
#
#     def forward(self, Z, tags) -> torch.Tensor:
#         """
#         Returns
#         -------
#         h: torch.Tensor, shape=(nAtoms, emb_size)
#             Atom embeddings.
#         """
#         h = torch.cat((self.embeddings(Z - 1), self.embeddings_tags(tags)),
#                       dim=1)  # -1 because Z.min()=1 (==Hydrogen)
#         return h


# z+distances+ads_features → dense → h
#     def __init__(self, emb_size: int, atom_features: dict) -> None:
#         super().__init__()
#         self.emb_size = emb_size
#         self.atom_features = atom_features
#
#         # Assuming each list in atom_features represents a feature vector for each atomic number
#         feature_size = len(next(iter(atom_features.values())))
#
#         # Adjusted feature size: feature_size from one-hot + 1 from center_distance + 74 from adstype
#         adjusted_feature_size = feature_size + 1 + 74
#
#         # Create a dense layer to transform the concatenated feature vector to the desired embedding size
#         self.dense = nn.Linear(adjusted_feature_size, emb_size)

#     def forward(self, Z, center_distance, adstype):
#         # Get the feature vectors for each atom
#         device = next(self.dense.parameters()).device
#         features = torch.stack([
#             torch.cat([
#                 torch.tensor(self.atom_features[z.item()], dtype=torch.float32).to(device),
#                 torch.tensor([cd], dtype=torch.float32).to(device),
#                 adt.to(device)  # Ensure `adt` is moved to the same device
#             ])
#             for z, cd, adt in zip(Z, center_distance, adstype)
#         ]).to(device)
#         # Apply the dense layer to adjust the dimensionality
#         h = self.dense(features)
#         return h

# z→ dense+distances+ads_features → h
#     def __init__(self, emb_size: int, atom_features: dict) -> None:
#         super().__init__()
#         self.emb_size = emb_size
#         self.atom_features = atom_features
#
#         # Assuming each list in atom_features represents a feature vector for each atomic number
#         feature_size = len(next(iter(atom_features.values())))
#
#         # Adjusted embedding size: emb_size - 75
#         intermediate_emb_size = emb_size - 75
#
#         # Create a dense layer to transform the feature vector to the intermediate embedding size
#         self.dense = nn.Linear(feature_size, intermediate_emb_size)
#
#
#     def forward(self, Z, distances, ads_features):
#         # Get the feature vectors for each atom
#         device = next(self.dense.parameters()).device
#         features = torch.stack([
#             torch.tensor(self.atom_features[z.item()], dtype=torch.float32).to(device)
#             for z in Z
#         ]).to(device)
#
#         # Apply the dense layer to adjust the dimensionality
#         transformed_features = self.dense(features)
#
#         # Ensure distances and ads_features are moved to the same device
#         distances = distances.to(device).unsqueeze(1)  # Ensure distances is 2D with shape [N, 1]
#         ads_features = ads_features.to(device)
#
#         # Concatenate the transformed features, distances, and ads_features
#         h = torch.cat([transformed_features, distances, ads_features], dim=-1)
#
#         return h
#
    # z+atoms_feature->h
    # def __init__(self, emb_size: int, num_elements:int) -> None:
    #     super().__init__()
    #     self.emb_size = emb_size
    #     emb_size = emb_size -  74
    #     self.atom_linear = nn.Linear(7, emb_size)
    #     for key in atom_info_normal:
    #         atom_info_normal[key] = torch.tensor(atom_info_normal[key])
    #
    #     self.atom_info_tensor = torch.cat([atom_info_normal[key] for key in atom_info_normal], dim=1)
    #     # self.atom_linear = nn.Embedding(num_elements, emb_size)
    #     # torch.nn.init.uniform_(self.atom_linear.weight,
    #     #                        a=-np.sqrt(3),
    #     #                        b=np.sqrt(3)
    #     #                        )
    # def forward(self, Z, ads_features):
    #     # Get the feature vectors for each atom
    #     device = next(self.atom_linear.parameters()).device
    #     self.atom_info_tensor = self.atom_info_tensor.to(device)
    #     atom_emb = self.atom_linear(self.atom_info_tensor)
    #     h = atom_emb[Z-1]
    #     # h = self.atom_linear(Z-1)
    #     # distance = distance.to(device).unsqueeze(1)
    #     ads_features = ads_features.to(device)
    #     # h_combined = torch.cat((h, distance, ads_features), dim=1).to(device)
    #     h_combined = torch.cat((h, ads_features), dim=1).to(device)
    #     # h_combined.requires_grad_(True)
    #     return h_combined

# 卷积
#     def __init__(self, emb_size: int, atom_features: dict) -> None:
#         super().__init__()
#         self.emb_size = emb_size
#         self.atom_features = atom_features
#
#         # Assuming each list in atom_features represents a feature vector for each atomic number
#         feature_size = len(next(iter(atom_features.values())))
#
#         # Adjusted embedding size: emb_size - 75
#         intermediate_emb_size = emb_size - 1
#         # intermediate_emb_size = emb_size
#
#         # Create a dense layer to transform the feature vector to the intermediate embedding size
#         self.dense = nn.Linear(feature_size, intermediate_emb_size)
#
#
#
#     def forward(self, Z, distances):
#     # def forward(self, Z):
#         # Get the feature vectors for each atom
#         device = next(self.dense.parameters()).device
#         features = torch.stack([
#             torch.tensor(self.atom_features[z.item()], dtype=torch.float32).to(device)
#             for z in Z
#         ]).to(device)
#
#         # Apply the dense layer to adjust the dimensionality
#         transformed_features = self.dense(features)
#
#         # Ensure distances and ads_features are moved to the same device
#         distances = distances.to(device).unsqueeze(1)  # Ensure distances is 2D with shape [N, 1]
#
#         # Concatenate the transformed features, distances, and ads_features
#         h = torch.cat([transformed_features, distances], dim=-1)
#         # h = transformed_features
#         return h


class EdgeEmbedding(torch.nn.Module):
    """
    Edge embedding based on the concatenation of atom embeddings
    and a subsequent dense layer.

    Arguments
    ---------
    atom_features: int
        Embedding size of the atom embedding.
    edge_features: int
        Embedding size of the input edge embedding.
    out_features: int
        Embedding size after the dense layer.
    activation: str
        Activation function used in the dense layer.
    """

    def __init__(
        self,
        atom_features: int,
        edge_features: int,
        out_features: int,
        activation: Optional[str] = None,
    ) -> None:
        super().__init__()
        in_features = 2 * atom_features + edge_features
        self.dense = Dense(
            in_features, out_features, activation=activation, bias=False
        )

    def forward(
        self,
        h: torch.Tensor,
        m: torch.Tensor,
        edge_index,
    ) -> torch.Tensor:
        """
        Arguments
        ---------
        h: torch.Tensor, shape (num_atoms, atom_features)
            Atom embeddings.
        m: torch.Tensor, shape (num_edges, edge_features)
            Radial basis in embedding block,
            edge embedding in interaction block.

        Returns
        -------
            m_st: torch.Tensor, shape=(nEdges, emb_size)
                Edge embeddings.
        """
        h_s = h[edge_index[0]]  # shape=(nEdges, emb_size)
        h_t = h[edge_index[1]]  # shape=(nEdges, emb_size)

        m_st = torch.cat(
            [h_s, h_t, m], dim=-1 # 边的原子特征向量+距离
        )  # (nEdges, 2*emb_size+nFeatures)
        m_st = self.dense(m_st)  # (nEdges, emb_size)
        return m_st


class AtomEmbeddingTags(torch.nn.Module):
    """
    Initial atom embeddings based on the atom type

    Arguments
    ---------
    emb_size: int
        Atom embeddings size
    """

# 增加原子标签
    def __init__(self, emb_size: int, num_elements: int) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.embeddings = torch.nn.Embedding(num_elements, emb_size-128)
        # init by uniform distribution
        torch.nn.init.uniform_(
            self.embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3)
        )
        self.embeddings_tags = torch.nn.Embedding(5, 128)
        # init by uniform distribution
        torch.nn.init.uniform_(
            self.embeddings_tags.weight, a=-np.sqrt(3), b=np.sqrt(3)
        )


    def forward(self, Z, tags) -> torch.Tensor:
        """
        Returns
        -------
        h: torch.Tensor, shape=(nAtoms, emb_size)
            Atom embeddings.
        """
        tags = tags.long()
        h = torch.cat((self.embeddings(Z - 1), self.embeddings_tags(tags)),
                      dim=1)  # -1 because Z.min()=1 (==Hydrogen)
        return h