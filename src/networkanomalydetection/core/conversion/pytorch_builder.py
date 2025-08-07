"""
Constructeur d'objets PyTorch Geometric depuis données extraites
"""
import torch
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class PyTorchGeometricBuilder:
    """Construction d'objets PyTorch Geometric Data"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    def build(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Construction de l'objet PyTorch Geometric Data
        
        Args:
            extracted_data: Données extraites par GraphDataExtractor
            
        Returns:
            Dict contenant Data object et métadonnées
        """
        logger.info("Construction de l'objet PyTorch Geometric")
        
        # Extraction des données
        node_features = extracted_data['node_features']
        edge_index = extracted_data['edge_index'] 
        edge_attr = extracted_data['edge_attr']
        
        # Validation des dimensions
        self._validate_dimensions(node_features, edge_index, edge_attr)
        
        # Conversion en tenseurs PyTorch
        torch_data = self._build_data_object(node_features, edge_index, edge_attr)
        
        # Métadonnées enrichies
        metadata = self._build_metadata(extracted_data, torch_data)
        
        return {
            'data': torch_data,
            'metadata': metadata,
            'node_mapping': extracted_data['node_mapping'],
            'reverse_mapping': extracted_data['reverse_mapping']
        }
    
    def _validate_dimensions(self, node_features: np.ndarray, 
                           edge_index: np.ndarray, edge_attr: np.ndarray):
        """Validation des dimensions des données"""
        
        num_nodes, node_dim = node_features.shape
        edge_dim_index, num_edges = edge_index.shape
        num_edges_attr, edge_dim = edge_attr.shape
        
        # Vérifications
        assert edge_dim_index == 2, f"edge_index doit avoir 2 lignes, trouvé {edge_dim_index}"
        assert num_edges == num_edges_attr, f"Incohérence nombre d'arêtes: {num_edges} vs {num_edges_attr}"
        assert node_dim == edge_dim, f"Dimensions embeddings incohérentes: nœuds {node_dim}D vs arêtes {edge_dim}D"
        
        # Vérification des indices
        max_node_idx = np.max(edge_index)
        assert max_node_idx < num_nodes, f"Index nœud {max_node_idx} >= nombre de nœuds {num_nodes}"
        
        logger.info(f"Validation OK: {num_nodes} nœuds, {num_edges} arêtes, {node_dim}D embeddings")
    
    def _build_data_object(self, node_features: np.ndarray, 
                          edge_index: np.ndarray, edge_attr: np.ndarray):
        """Construction de l'objet Data PyTorch Geometric"""
        
        # Conversion en tenseurs PyTorch avec types optimaux
        x = torch.tensor(node_features, dtype=torch.float32, device=self.device)
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32, device=self.device)
        
        # Construction du Data object
        # Note: Utilisation d'un dict au lieu de torch_geometric.data.Data pour éviter la dépendance
        torch_data = {
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'num_nodes': x.size(0),
            'num_edges': edge_index.size(1)
        }
        
        logger.info(f"Data object créé: x={x.shape}, edge_index={edge_index.shape}, edge_attr={edge_attr.shape}")
        return torch_data
    
    def _build_metadata(self, extracted_data: Dict, torch_data: Dict) -> Dict:
        """Construction des métadonnées enrichies"""
        
        metadata = {
            'conversion_info': {
                'source': 'NetworkX MultiDiGraph',
                'target': 'PyTorch Geometric',
                'timestamp': torch.utils.data.dataset._get_current_time() if hasattr(torch.utils.data.dataset, '_get_current_time') else 'unknown',
                'device': self.device
            },
            'tensor_info': {
                'x_shape': list(torch_data['x'].shape),
                'x_dtype': str(torch_data['x'].dtype),
                'edge_index_shape': list(torch_data['edge_index'].shape),
                'edge_index_dtype': str(torch_data['edge_index'].dtype),
                'edge_attr_shape': list(torch_data['edge_attr'].shape),
                'edge_attr_dtype': str(torch_data['edge_attr'].dtype)
            },
            'statistics': {
                'num_nodes': torch_data['num_nodes'],
                'num_edges': torch_data['num_edges'],
                'avg_degree': torch_data['num_edges'] * 2.0 / torch_data['num_nodes'],
                'node_feature_dim': torch_data['x'].size(1),
                'edge_feature_dim': torch_data['edge_attr'].size(1)
            },
            'original_metadata': extracted_data.get('global_metadata', {}),
            'node_metadata': extracted_data.get('node_metadata', {}),
            'edge_metadata': extracted_data.get('edge_metadata', {})
        }
        
        return metadata
