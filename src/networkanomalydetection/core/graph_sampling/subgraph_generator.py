"""
Subgraph Sampler pour entraînement GNN
Génération séquentielle par packet_id avec limitation de taille
CORRECTION: Ego-graph centré uniquement sur nœuds centraux (node_type=1)
"""
import networkx as nx
import numpy as np
import random
import logging
from typing import List, Dict, Any, Iterator, Tuple
from tqdm import tqdm

logger = logging.getLogger(__name__)

class SubgraphSampler:
    """
    Générateur de subgraphs pour entraînement GNN
    Stratégie: packet_id séquentiel + ego-graph profondeur 2 
    CENTRÉ UNIQUEMENT SUR NŒUDS CENTRAUX
    """
    
    def __init__(self, networkx_graph: nx.MultiDiGraph,
                 radius: int = 2):
        
        self.graph = networkx_graph
        self.radius = radius
        
        
        # Extraire packet_ids triés
        self.packet_ids = self._extract_packet_ids()
        
        logger.info(f"Subgraph Sampler: {len(self.packet_ids)} packet_ids, radius={radius}")
    
    def _extract_packet_ids(self) -> List[int]:
        """Extraire packet_ids uniques triés"""
        packet_ids = set()
        for node_id, attrs in self.graph.nodes(data=True):
            packet_id = attrs.get('packet_id')
            if packet_id is not None:
                packet_ids.add(packet_id)
        return sorted(list(packet_ids))
    
    def _get_valid_nodes(self, max_packet_id: int) -> List[str]:
        """Filtrage temporal: packet_id <= max_packet_id"""
        valid_nodes = []
        for node_id, attrs in self.graph.nodes(data=True):
            packet_id = attrs.get('packet_id')
            if packet_id is not None and packet_id <= max_packet_id:
                valid_nodes.append(node_id)
        return valid_nodes
    
    def _get_target_nodes(self, packet_id: int) -> List[str]:
        """Nœuds CENTRAUX avec packet_id exact (node_type=1 uniquement)"""
        target_nodes = []
        for node_id, attrs in self.graph.nodes(data=True):
            if (attrs.get('packet_id') == packet_id and 
                attrs.get('node_type') == 1):  # SEULEMENT nœuds centraux
                target_nodes.append(node_id)
        return target_nodes
    
    def _generate_ego_graph(self, target_node: str, max_packet_id: int) -> nx.Graph:
        """Générer ego-graph taille naturelle"""
        
        # Filtrage temporal
        valid_nodes = self._get_valid_nodes(max_packet_id)
        filtered_graph = self.graph.subgraph(valid_nodes)
        
        if target_node not in filtered_graph:
            return None
        
        try:
            # Ego-graph 
            ego_graph = nx.ego_graph(filtered_graph, target_node, radius=self.radius)
            return ego_graph
            
        except Exception as e:
            logger.debug(f"Erreur ego-graph {target_node}: {e}")
            return None
    
    def _is_valid_subgraph(self, subgraph: nx.Graph) -> bool:
        """Validation subgraph logique"""
        if subgraph is None:
            return False
        
        num_nodes = subgraph.number_of_nodes()
        num_edges = subgraph.number_of_edges()
        
        # Critères minimaux
        if num_nodes < 2:
            return False
        
        # Pour 2+ nœuds, au moins 1 arête (structure connectée)
        return num_edges >= 1
    
    def generate_batch(self, packet_id: int) -> List[nx.Graph]:
        """Générer batch de subgraphs pour un packet_id"""
        
        target_nodes = self._get_target_nodes(packet_id)
        
        if not target_nodes:
            return []
        
        # Prendre TOUS les nœuds centraux (pas de limitation)
        valid_subgraphs = []
        for target_node in target_nodes:
            subgraph = self._generate_ego_graph(target_node, packet_id)
            if self._is_valid_subgraph(subgraph):
                valid_subgraphs.append(subgraph)
        
        return valid_subgraphs
    
    def batch_iterator(self, show_progress: bool = True) -> Iterator[Tuple[int, List[nx.Graph]]]:
        """Itérateur principal avec barre de progression"""
        packet_iter = tqdm(self.packet_ids, 
                          desc="Processing packets",
                          unit="packet",
                          disable=not show_progress) if show_progress else self.packet_ids
        
        total_subgraphs = 0
        valid_packets = 0
        
        for packet_id in packet_iter:
            subgraphs = self.generate_batch(packet_id)
            if subgraphs:
                total_subgraphs += len(subgraphs)
                valid_packets += 1
                
                # Mise à jour description tqdm
                if show_progress:
                    packet_iter.set_postfix({
                        'valid_packets': valid_packets,
                        'total_subgraphs': total_subgraphs,
                        'avg_per_packet': f"{total_subgraphs/valid_packets:.1f}"
                    })
                
                yield packet_id, subgraphs

class BatchProcessor:
    """Conversion NetworkX vers PyTorch"""
    
    def __init__(self):
        from networkanomalydetection.core.graph_vectorization import NetworkXToTorchConverter
        self.converter = NetworkXToTorchConverter(validate=False, device='cpu')
    
    def process_batch(self, subgraphs: List[nx.Graph]) -> List[Dict[str, Any]]:
        """Convertir batch NetworkX vers PyTorch"""
        pytorch_subgraphs = []
        
        for subgraph in subgraphs:
            try:
                # Gestion subgraphs sans arêtes
                if subgraph.number_of_edges() == 0 and subgraph.number_of_nodes() >= 2:
                    # Ajouter arête temporaire pour conversion
                    temp_graph = subgraph.copy()
                    nodes = list(temp_graph.nodes())
                    temp_graph.add_edge(nodes[0], nodes[1], 
                                      label='temp',
                                      embedding=np.zeros(64, dtype=np.float32),
                                      entity_type='TEMP',
                                      classification_confidence=0.5)
                    
                    pytorch_data, _, _ = self.converter.convert(temp_graph)
                    pytorch_data['synthetic_edge'] = True
                else:
                    pytorch_data, _, _ = self.converter.convert(subgraph)
                    pytorch_data['synthetic_edge'] = False
                
                pytorch_subgraphs.append(pytorch_data)
                
            except Exception as e:
                logger.debug(f"Erreur conversion: {e}")
                continue
        
        return pytorch_subgraphs
    
    def process_iterator(self, batch_iterator: Iterator[Tuple[int, List[nx.Graph]]], 
                        show_progress: bool = True) -> Iterator[Tuple[int, List[Dict]]]:
        """Processeur d'itérateur avec progression"""
        
        processed = 0
        failed = 0
        
        for packet_id, nx_subgraphs in batch_iterator:
            pytorch_subgraphs = self.process_batch(nx_subgraphs)
            
            if pytorch_subgraphs:
                processed += len(pytorch_subgraphs)
                yield packet_id, pytorch_subgraphs
            else:
                failed += 1
            
            # Log périodique
            if show_progress and (processed + failed) % 1000 == 0:
                logger.info(f"Processed: {processed} subgraphs, Failed: {failed}")


def create_training_pipeline(vectorized_graph: nx.MultiDiGraph,
                           radius: int = 2,
                           show_progress: bool = True) -> Iterator[Tuple[int, List[Dict]]]:
    """Pipeline complet génération + conversion avec progression"""
    
    sampler = SubgraphSampler(vectorized_graph, radius)
    processor = BatchProcessor()
    
    logger.info(f"Starting pipeline: {len(sampler.packet_ids)} packets to process")
    
    batch_iterator = sampler.batch_iterator(show_progress=show_progress)
    return processor.process_iterator(batch_iterator, show_progress=show_progress)