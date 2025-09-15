"""
Architecture GNN Auto-encodeur pour détection d'anomalies tri-domaine
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv

from .heads import EdgeAnomalyHead, MultiNodeAnomalyHead


class EnhancedGNNAutoEncoder(nn.Module):
    """
    GNN Auto-encodeur avec détection tri-domaine complète :
    - Détection nœuds centraux
    - Détection nœuds paramètres
    - Détection edges
    """

    def __init__(self,
                 input_dim: int = 64,
                 hidden_dims: list = [128, 256, 128],
                 edge_dim: int = 64,
                 dropout: float = 0.2):
        super().__init__()

        # GraphSAGE backbone
        self.convs = nn.ModuleList()
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            self.convs.append(SAGEConv(dims[i], dims[i+1], aggr='mean'))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        # Heads spécialisés tri-domaine
        final_dim = hidden_dims[-1]
        self.node_head = MultiNodeAnomalyHead(input_dim=final_dim, output_dim=input_dim)
        self.edge_head = EdgeAnomalyHead(node_dim=final_dim, edge_dim=edge_dim)

    def forward(self, data):
        """
        Args:
            data: PyTorch Geometric Data object
        Returns:
            dict avec reconstructions tri-domaine
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # GraphSAGE backbone
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = self.activation(h)
            h = self.dropout(h)

        # Récupérer node_types
        node_types = getattr(data, 'node_type', torch.ones(x.size(0), dtype=torch.long))

        # Reconstructions tri-domaine
        node_reconstructions, node_masks = self.node_head(h, node_types)
        edge_reconstruction = self.edge_head(h, edge_index, edge_attr)

        return {
            'central_reconstruction': node_reconstructions['central'],
            'param_reconstruction': node_reconstructions['param'],
            'edge_reconstruction': edge_reconstruction,
            'node_embeddings': h,
            'node_types': node_types,
            'central_mask': node_masks['central_mask'],
            'param_mask': node_masks['param_mask']
        }

    def compute_loss(self, outputs, data, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0):
        """
        Calcul de la loss tri-domaine

        Args:
            outputs: Résultats du forward
            data: Données originales
            alpha: Poids loss nœuds centraux
            beta: Poids loss nœuds paramètres
            gamma: Poids loss edges
        """
        losses = {}
        total_loss = 0

        # Loss nœuds centraux
        if outputs['central_reconstruction'].size(0) > 0:
            original_central = data.x[outputs['central_mask']]
            central_loss = F.mse_loss(outputs['central_reconstruction'], original_central)
            losses['central_loss'] = central_loss
            total_loss += alpha * central_loss
        else:
            losses['central_loss'] = torch.tensor(0.0)

        # Loss nœuds paramètres
        if outputs['param_reconstruction'].size(0) > 0:
            original_param = data.x[outputs['param_mask']]
            param_loss = F.mse_loss(outputs['param_reconstruction'], original_param)
            losses['param_loss'] = param_loss
            total_loss += beta * param_loss
        else:
            losses['param_loss'] = torch.tensor(0.0)

        # Loss edges
        edge_loss = F.mse_loss(outputs['edge_reconstruction'], data.edge_attr)
        losses['edge_loss'] = edge_loss
        total_loss += gamma * edge_loss

        losses['total_loss'] = total_loss
        return losses

    def compute_anomaly_scores(self, outputs, data):
        """
        Calcul des scores d'anomalie tri-domaine
        """
        scores = {}

        # Scores nœuds centraux
        if outputs['central_reconstruction'].size(0) > 0:
            original_central = data.x[outputs['central_mask']]
            central_errors = F.mse_loss(
                outputs['central_reconstruction'],
                original_central,
                reduction='none'
            ).mean(dim=1)
            scores['central_anomaly_scores'] = central_errors
        else:
            scores['central_anomaly_scores'] = torch.tensor([])

        # Scores nœuds paramètres
        if outputs['param_reconstruction'].size(0) > 0:
            original_param = data.x[outputs['param_mask']]
            param_errors = F.mse_loss(
                outputs['param_reconstruction'],
                original_param,
                reduction='none'
            ).mean(dim=1)
            scores['param_anomaly_scores'] = param_errors
        else:
            scores['param_anomaly_scores'] = torch.tensor([])

        # Scores edges
        edge_errors = F.mse_loss(
            outputs['edge_reconstruction'],
            data.edge_attr,
            reduction='none'
        ).mean(dim=1)
        scores['edge_anomaly_scores'] = edge_errors

        return scores
