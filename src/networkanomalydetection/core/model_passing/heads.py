"""
Heads améliorés pour détection multi-domaine complète
"""
import torch
from torch import nn


class MultiNodeAnomalyHead(nn.Module):
    """Détection d'anomalies pour nœuds centraux ET paramètres"""

    def __init__(self, input_dim: int = 128, output_dim: int = 64, hidden_dim: int = 64):
        super().__init__()

        # Heads spécialisés par type de nœud
        self.central_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

        self.param_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, node_embeddings, node_types):
        """
        Args:
            node_embeddings: [N, input_dim] embeddings de tous les nœuds
            node_types: [N] types des nœuds (0=param, 1=central)
        Returns:
            dict avec reconstructions par type de nœud
        """
        reconstructions = {}

        # Nœuds centraux (node_type=1)
        central_mask = (node_types == 1)
        if central_mask.sum() > 0:
            central_embeddings = node_embeddings[central_mask]
            reconstructions['central'] = self.central_head(central_embeddings)
        else:
            reconstructions['central'] = torch.empty(0, self.central_head[-1].out_features)

        # Nœuds paramètres (node_type=0)
        param_mask = (node_types == 2)
        if param_mask.sum() > 0:
            param_embeddings = node_embeddings[param_mask]
            reconstructions['param'] = self.param_head(param_embeddings)
        else:
            reconstructions['param'] = torch.empty(0, self.param_head[-1].out_features)

        return reconstructions, {'central_mask': central_mask, 'param_mask': param_mask}


class EdgeAnomalyHead(nn.Module):
    """Détection d'anomalies pour edges (inchangé)"""

    def __init__(self, node_dim: int = 128, edge_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        input_dim = 2 * node_dim + edge_dim

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, edge_dim)
        )

    def forward(self, node_embeddings, edge_index, edge_features):
        src_embeddings = node_embeddings[edge_index[0]]
        dst_embeddings = node_embeddings[edge_index[1]]

        edge_input = torch.cat([
            src_embeddings,
            dst_embeddings,
            edge_features
        ], dim=1)

        return self.classifier(edge_input)
