"""
Extracteur de données NetworkX pour conversion PyTorch Geometric
"""
import numpy as np
import networkx as nx
import logging
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

class GraphDataExtractor:
    """Extraction des données depuis NetworkX MultiDiGraph"""
    
    def __init__(self):
        self.node_mapping = {}
        self.reverse_mapping = {}
    
    def extract(self, nx_graph: nx.MultiDiGraph) -> Dict[str, Any]:
        """
        Extraction complète des données du graphe
        
        Args:
            nx_graph: Graphe NetworkX vectorisé
            
        Returns:
            Dict contenant node_features, edge_data et metadata
        """
        logger.info(f"Extraction des données du graphe: {nx_graph.number_of_nodes()} nœuds, {nx_graph.number_of_edges()} arêtes")
        
        # Extraction des nœuds
        node_features, node_metadata = self._extract_node_features(nx_graph)
        
        # Extraction des arêtes  
        edge_index, edge_attr, edge_metadata = self._extract_edge_data(nx_graph)
        
        # Métadonnées globales
        global_metadata = self._extract_global_metadata(nx_graph)
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'node_metadata': node_metadata,
            'edge_metadata': edge_metadata,
            'global_metadata': global_metadata,
            'node_mapping': self.node_mapping.copy(),
            'reverse_mapping': self.reverse_mapping.copy()
        }
    
    def _extract_node_features(self, nx_graph: nx.MultiDiGraph) -> Tuple[np.ndarray, Dict]:
        """Extraction des features des nœuds"""
        
        node_features = []
        node_metadata = {
            'entity_types': [],
            'classification_confidences': [],
            'node_types': [],
            'packet_ids': [],
            'original_labels': []
        }
        
        # Créer le mapping des IDs
        self.node_mapping = {node_id: idx for idx, node_id in enumerate(nx_graph.nodes())}
        self.reverse_mapping = {idx: node_id for node_id, idx in self.node_mapping.items()}
        
        # Extraire features et métadonnées
        for node_id, attrs in nx_graph.nodes(data=True):
            # Features principales (embeddings)
            if 'embedding' in attrs:
                embedding = attrs['embedding']
                if isinstance(embedding, np.ndarray):
                    node_features.append(embedding)
                else:
                    node_features.append(np.array(embedding, dtype=np.float32))
            else:
                # Fallback si pas d'embedding
                logger.warning(f"Nœud {node_id} sans embedding, utilisation de zéros")
                node_features.append(np.zeros(64, dtype=np.float32))
            
            # Métadonnées
            node_metadata['entity_types'].append(attrs.get('entity_type', 'UNKNOWN'))
            node_metadata['classification_confidences'].append(attrs.get('classification_confidence', 0.0))
            node_metadata['node_types'].append(attrs.get('node_type', 0))
            node_metadata['packet_ids'].append(attrs.get('packet_id', 0))
            node_metadata['original_labels'].append(attrs.get('label', ''))
        
        node_features = np.array(node_features, dtype=np.float32)
        
        logger.info(f"Features des nœuds extraites: {node_features.shape}")
        return node_features, node_metadata
    
    def _extract_edge_data(self, nx_graph: nx.MultiDiGraph) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Extraction des données d'arêtes"""
        
        edge_connections = []
        edge_features = []
        edge_metadata = {
            'edge_types': [],
            'entity_types': [],
            'classification_confidences': [],
            'original_labels': []
        }
        
        # Parcourir toutes les arêtes (avec clés pour MultiDiGraph)
        for u, v, key, attrs in nx_graph.edges(data=True, keys=True):
            # Conversion des IDs vers indices
            u_idx = self.node_mapping[u]
            v_idx = self.node_mapping[v]
            edge_connections.append([u_idx, v_idx])
            
            # Features des arêtes (embeddings)
            if 'embedding' in attrs:
                embedding = attrs['embedding']
                if isinstance(embedding, np.ndarray):
                    edge_features.append(embedding)
                else:
                    edge_features.append(np.array(embedding, dtype=np.float32))
            else:
                # Fallback si pas d'embedding
                logger.warning(f"Arête {u}->{v} (key={key}) sans embedding")
                edge_features.append(np.zeros(64, dtype=np.float32))
            
            # Métadonnées des arêtes
            edge_metadata['edge_types'].append(attrs.get('label', 'unknown'))
            edge_metadata['entity_types'].append(attrs.get('entity_type', 'UNKNOWN'))
            edge_metadata['classification_confidences'].append(attrs.get('classification_confidence', 0.0))
            edge_metadata['original_labels'].append(attrs.get('label', ''))
        
        # Conversion en arrays NumPy
        edge_index = np.array(edge_connections, dtype=np.int64).T  # Format [2, num_edges]
        edge_attr = np.array(edge_features, dtype=np.float32)
        
        logger.info(f"Données d'arêtes extraites: edge_index {edge_index.shape}, edge_attr {edge_attr.shape}")
        return edge_index, edge_attr, edge_metadata
    
    def _extract_global_metadata(self, nx_graph: nx.MultiDiGraph) -> Dict:
        """Extraction des métadonnées globales"""
        
        # Statistiques de base
        num_nodes = nx_graph.number_of_nodes()
        num_edges = nx_graph.number_of_edges()
        
        # Analyse des degrés
        degrees = [nx_graph.degree(node) for node in nx_graph.nodes()]
        in_degrees = [nx_graph.in_degree(node) for node in nx_graph.nodes()]
        out_degrees = [nx_graph.out_degree(node) for node in nx_graph.nodes()]
        
        metadata = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'is_directed': nx_graph.is_directed(),
            'is_multigraph': nx_graph.is_multigraph(),
            'degree_stats': {
                'mean': np.mean(degrees),
                'std': np.std(degrees),
                'min': np.min(degrees),
                'max': np.max(degrees)
            },
            'in_degree_stats': {
                'mean': np.mean(in_degrees),
                'std': np.std(in_degrees),
                'min': np.min(in_degrees),
                'max': np.max(in_degrees)
            },
            'out_degree_stats': {
                'mean': np.mean(out_degrees),
                'std': np.std(out_degrees),
                'min': np.min(out_degrees),
                'max': np.max(out_degrees)
            },
            'density': nx.density(nx_graph)
        }
        
        return metadata