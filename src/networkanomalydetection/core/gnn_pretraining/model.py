"""
Modèle GNN Multi-Domaines pour Pré-entraînement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from collections import defaultdict, deque
import time

logger = logging.getLogger(__name__)

class DynamicNodeMemory:
    """Mémoire dynamique pour les nœuds partagés"""
    
    def __init__(self, window_size: int = 300):
        self.window_size = window_size
        self.node_timeline = defaultdict(lambda: deque(maxlen=1000))
        self.features_cache = {}
        self.last_cleanup = time.time()
    
    def update_node_activity(self, node_id: str, timestamp: float, context_features: torch.Tensor):
        """Met à jour l'activité d'un nœud"""
        self.node_timeline[node_id].append((timestamp, context_features.clone().detach()))
        
        if node_id in self.features_cache:
            del self.features_cache[node_id]
        
        current_time = time.time()
        if current_time - self.last_cleanup > 60.0:
            self._cleanup_old_entries(current_time)
            self.last_cleanup = current_time
    
    def _cleanup_old_entries(self, current_timestamp: float):
        """Nettoie les entrées anciennes"""
        cutoff_time = current_timestamp - self.window_size
        
        for node_id in list(self.node_timeline.keys()):
            timeline = self.node_timeline[node_id]
            while timeline and timeline[0][0] < cutoff_time:
                timeline.popleft()
            if not timeline:
                del self.node_timeline[node_id]
        
        self.features_cache.clear()
    
    def get_dynamic_features(self, node_id: str, current_timestamp: float) -> torch.Tensor:
        """Calcule les features dynamiques pour un nœud"""
        
        cache_key = f"{node_id}_{int(current_timestamp)}"
        if cache_key in self.features_cache:
            return self.features_cache[cache_key]
        
        if node_id not in self.node_timeline:
            features = torch.zeros(8, dtype=torch.float32)
            self.features_cache[cache_key] = features
            return features
        
        timeline = self.node_timeline[node_id]
        if not timeline:
            features = torch.zeros(8, dtype=torch.float32)
            self.features_cache[cache_key] = features
            return features
        
        cutoff_time = current_timestamp - self.window_size
        recent_activities = [(ts, ctx) for ts, ctx in timeline if ts >= cutoff_time]
        
        if not recent_activities:
            features = torch.zeros(8, dtype=torch.float32)
            self.features_cache[cache_key] = features
            return features
        
        features = self._calculate_dynamic_features(recent_activities, current_timestamp)
        self.features_cache[cache_key] = features
        return features
    
    def _calculate_dynamic_features(self, activities: List[Tuple[float, torch.Tensor]], 
                                   current_timestamp: float) -> torch.Tensor:
        """Calcule les features dynamiques"""
        
        if not activities:
            return torch.zeros(8, dtype=torch.float32)
        
        timestamps = [act[0] for act in activities]
        contexts = [act[1] for act in activities]
        
        # Feature 1: Fréquence
        frequency = len(activities) / (self.window_size / 60.0)
        frequency_normalized = min(frequency / 10.0, 1.0)
        
        # Feature 2: Accélération
        if len(timestamps) >= 2:
            recent_interval = timestamps[-1] - timestamps[-2]
            acceleration = 1.0 / max(recent_interval, 1.0)
        else:
            acceleration = 0.0
        
        # Feature 3: Régularité
        if len(timestamps) >= 3:
            intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
            regularity = 1.0 / (1.0 + np.std(intervals))
        else:
            regularity = 0.5
        
        # Feature 4: Concentration temporelle
        recent_threshold = current_timestamp - (self.window_size * 0.1)
        recent_count = sum(1 for ts in timestamps if ts >= recent_threshold)
        concentration = recent_count / len(timestamps)
        
        # Feature 5: Persistance
        if len(timestamps) >= 2:
            persistence = (timestamps[-1] - timestamps[0]) / self.window_size
        else:
            persistence = 0.0
        
        # Feature 6: Diversité contextuelle
        if contexts and len(contexts) > 1:
            context_stack = torch.stack(contexts)
            context_diversity = torch.var(context_stack, dim=0).mean().item()
        else:
            context_diversity = 0.0
        
        # Feature 7: Nouveauté
        time_since_first = current_timestamp - timestamps[0]
        novelty = 1.0 / (1.0 + time_since_first / 3600.0)
        
        # Feature 8: Intensité récente
        very_recent_threshold = current_timestamp - (self.window_size * 0.25)
        very_recent_count = sum(1 for ts in timestamps if ts >= very_recent_threshold)
        intensity = very_recent_count / len(timestamps)
        
        return torch.tensor([
            frequency_normalized, acceleration, regularity, concentration,
            persistence, context_diversity, novelty, intensity
        ], dtype=torch.float32)

class EdgeLevelGNN(nn.Module):
    """Niveau 1: Apprentissage des triplets normaux"""
    
    def __init__(self, node_dim: int = 64, edge_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        
        # Encodeur triplets
        self.triplet_encoder = nn.Sequential(
            nn.Linear(node_dim + edge_dim + node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # Décodeur pour reconstruction
        self.triplet_decoder = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim + edge_dim + node_dim)
        )
        
        # Score de normalité
        self.normality_scorer = nn.Sequential(
            nn.Linear(hidden_dim // 4, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_index: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass niveau 1"""
        
        source_embeddings = x[edge_index[0]]
        target_embeddings = x[edge_index[1]]
        
        triplets = torch.cat([source_embeddings, edge_attr, target_embeddings], dim=1)
        
        encoded = self.triplet_encoder(triplets)
        reconstructed = self.triplet_decoder(encoded)
        normality_scores = self.normality_scorer(encoded)
        
        return {
            'encoded_triplets': encoded,
            'reconstructed_triplets': reconstructed,
            'original_triplets': triplets,
            'normality_scores': normality_scores
        }

class NodeLevelGNN(MessagePassing):
    """Niveau 2: Apprentissage des paquets normaux"""
    
    def __init__(self, node_dim: int = 64, edge_dim: int = 64, hidden_dim: int = 128):
        super().__init__(aggr='mean')
        
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.node_reconstructor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
        self.normality_scorer = nn.Sequential(
            nn.Linear(hidden_dim // 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_index: torch.Tensor, 
                edge_features: torch.Tensor, node_types: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass niveau 2"""
        
        updated_embeddings = self.propagate(edge_index, x=x, edge_attr=edge_attr, 
                                          edge_features=edge_features)
        
        central_mask = (node_types == 1)
        central_indices = torch.where(central_mask)[0]
        
        if len(central_indices) == 0:
            return {
                'encoded_nodes': torch.empty(0, self.update_mlp[-1].out_features),
                'reconstructed_nodes': torch.empty(0, x.size(1)),
                'original_nodes': torch.empty(0, x.size(1)),
                'normality_scores': torch.empty(0, 1)
            }
        
        central_embeddings = x[central_indices]
        central_updated = updated_embeddings[central_indices]
        
        encoded_nodes = self.update_mlp(torch.cat([central_embeddings, central_updated], dim=1))
        reconstructed_nodes = self.node_reconstructor(encoded_nodes)
        normality_scores = self.normality_scorer(encoded_nodes)
        
        return {
            'encoded_nodes': encoded_nodes,
            'reconstructed_nodes': reconstructed_nodes,
            'original_nodes': central_embeddings,
            'normality_scores': normality_scores
        }
    
    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        """Construire les messages"""
        message_input = torch.cat([x_j, edge_attr], dim=1)
        message = self.message_mlp(message_input)
        
        if edge_features.size(1) == message.size(1):
            message = message + edge_features
        
        return message

class SharedNodeGNN(nn.Module):
    """Niveau 3: Apprentissage des nœuds partagés normaux"""
    
    def __init__(self, node_dim: int = 64, dynamic_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        
        self.dynamic_encoder = nn.Sequential(
            nn.Linear(dynamic_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(node_dim + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.shared_reconstructor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim + dynamic_dim)
        )
        
        self.normality_scorer = nn.Sequential(
            nn.Linear(hidden_dim // 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, shared_node_embeddings: torch.Tensor, 
                dynamic_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass niveau 3"""
        
        if len(shared_node_embeddings) == 0:
            return {
                'encoded_shared': torch.empty(0, self.fusion_layer[-1].out_features),
                'reconstructed_shared': torch.empty(0, shared_node_embeddings.size(1) + dynamic_features.size(1)),
                'original_shared': torch.empty(0, shared_node_embeddings.size(1) + dynamic_features.size(1)),
                'normality_scores': torch.empty(0, 1)
            }
        
        encoded_dynamic = self.dynamic_encoder(dynamic_features)
        combined_features = torch.cat([shared_node_embeddings, encoded_dynamic], dim=1)
        encoded_shared = self.fusion_layer(combined_features)
        
        original_combined = torch.cat([shared_node_embeddings, dynamic_features], dim=1)
        reconstructed_shared = self.shared_reconstructor(encoded_shared)
        normality_scores = self.normality_scorer(encoded_shared)
        
        return {
            'encoded_shared': encoded_shared,
            'reconstructed_shared': reconstructed_shared,
            'original_shared': original_combined,
            'normality_scores': normality_scores
        }

class MultiDomainPretrainingGNN(nn.Module):
    """Modèle GNN complet pour pré-entraînement"""
    
    def __init__(self, node_dim: int = 64, edge_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        
        self.edge_level_gnn = EdgeLevelGNN(node_dim, edge_dim, hidden_dim)
        self.node_level_gnn = NodeLevelGNN(node_dim, edge_dim, hidden_dim)
        self.shared_node_gnn = SharedNodeGNN(node_dim, dynamic_dim=8, hidden_dim=hidden_dim)
        
        self.dynamic_memory = DynamicNodeMemory(window_size=300)
        
        self.level_weights = nn.Parameter(torch.tensor([0.4, 0.4, 0.2]), requires_grad=True)
    
    def forward(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass complet"""
        
        x = batch_data['node_embeddings']
        edge_attr = batch_data['edge_embeddings'] 
        edge_index = batch_data['edge_index']
        node_types = batch_data['node_types']
        timestamps = batch_data.get('timestamps', torch.tensor([]))
        
        # Niveau 1: Arêtes
        edge_results = self.edge_level_gnn(x, edge_attr, edge_index)
        
        # Niveau 2: Nœuds centraux
        node_results = self.node_level_gnn(
            x, edge_attr, edge_index, 
            edge_results['encoded_triplets'], 
            node_types
        )
        
        # Niveau 3: Nœuds partagés
        shared_results = {}
        
        if 'shared_node_info' in batch_data and batch_data['shared_node_info'] is not None:
            shared_info = batch_data['shared_node_info']
            
            self._update_dynamic_memory(shared_info, timestamps)
            dynamic_features = self._compute_dynamic_features(shared_info, timestamps)
            
            shared_results = self.shared_node_gnn(
                shared_info['embeddings'],
                dynamic_features
            )
        else:
            shared_results = {
                'encoded_shared': torch.empty(0, 64),
                'reconstructed_shared': torch.empty(0, 72),
                'original_shared': torch.empty(0, 72),
                'normality_scores': torch.empty(0, 1)
            }
        
        return {
            **edge_results,
            **node_results,
            **shared_results,
            'level_weights': F.softmax(self.level_weights, dim=0)
        }
    
    def _update_dynamic_memory(self, shared_info: Dict, timestamps: torch.Tensor):
        """Met à jour la mémoire dynamique"""
        
        shared_node_ids = shared_info.get('node_ids', [])
        embeddings = shared_info.get('embeddings', torch.tensor([]))
        
        if len(timestamps) > 0:
            current_time = float(timestamps.mean().item())
        else:
            current_time = time.time()
        
        for i, node_id in enumerate(shared_node_ids):
            if i < len(embeddings):
                self.dynamic_memory.update_node_activity(
                    str(node_id), 
                    current_time,
                    embeddings[i]
                )
    
    def _compute_dynamic_features(self, shared_info: Dict, timestamps: torch.Tensor) -> torch.Tensor:
        """Calcule les features dynamiques"""
        
        shared_node_ids = shared_info.get('node_ids', [])
        
        if len(timestamps) > 0:
            current_time = float(timestamps.mean().item())
        else:
            current_time = time.time()
        
        dynamic_features = []
        for node_id in shared_node_ids:
            features = self.dynamic_memory.get_dynamic_features(str(node_id), current_time)
            dynamic_features.append(features)
        
        if dynamic_features:
            return torch.stack(dynamic_features)
        else:
            return torch.empty(0, 8)
    
    def compute_pretraining_loss(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Calcule la loss de pré-entraînement"""
        
        losses = {}
        total_loss = torch.tensor(0.0, requires_grad=True)
        
        weights = F.softmax(self.level_weights, dim=0)
        
        # Loss niveau 1: Arêtes
        if len(outputs['encoded_triplets']) > 0:
            edge_recon_loss = F.mse_loss(
                outputs['reconstructed_triplets'], 
                outputs['original_triplets']
            )
            edge_normality_loss = F.binary_cross_entropy(
                outputs['normality_scores'],
                torch.ones_like(outputs['normality_scores'])
            )
            edge_loss = edge_recon_loss + 0.5 * edge_normality_loss
            losses['edge_loss'] = edge_loss
            total_loss = total_loss + weights[0] * edge_loss
        
        # Loss niveau 2: Nœuds
        if len(outputs['encoded_nodes']) > 0:
            node_recon_loss = F.mse_loss(
                outputs['reconstructed_nodes'],
                outputs['original_nodes']
            )
            node_normality_loss = F.binary_cross_entropy(
                outputs['normality_scores'],
                torch.ones_like(outputs['normality_scores'])
            )
            node_loss = node_recon_loss + 0.5 * node_normality_loss
            losses['node_loss'] = node_loss
            total_loss = total_loss + weights[1] * node_loss
        
        # Loss niveau 3: Nœuds partagés
        if len(outputs['encoded_shared']) > 0:
            shared_recon_loss = F.mse_loss(
                outputs['reconstructed_shared'],
                outputs['original_shared']
            )
            shared_normality_loss = F.binary_cross_entropy(
                outputs['shared_normality_scores'],
                torch.ones_like(outputs['shared_normality_scores'])
            )
            shared_loss = shared_recon_loss + 0.5 * shared_normality_loss
            losses['shared_loss'] = shared_loss
            total_loss = total_loss + weights[2] * shared_loss
        
        losses['total_loss'] = total_loss
        losses['level_weights'] = weights
        
        return losses
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Statistiques mémoire"""
        return {
            'active_nodes': len(self.dynamic_memory.node_timeline),
            'total_entries': sum(len(timeline) for timeline in self.dynamic_memory.node_timeline.values()),
            'cache_size': len(self.dynamic_memory.features_cache)
        }
    
    def clear_memory(self):
        """Nettoie la mémoire"""
        self.dynamic_memory = DynamicNodeMemory(window_size=self.dynamic_memory.window_size)