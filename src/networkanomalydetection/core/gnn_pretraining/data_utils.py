"""
Utilitaires pour traitement des données du graphe unique

"""

import torch
import networkx as nx
from typing import Dict, List, Tuple, Any
import logging
import numpy as np
from collections import defaultdict
import time

logger = logging.getLogger(__name__)

def prepare_graph_data(vectorized_graph: nx.MultiDiGraph, 
                      max_nodes: int = None) -> Dict[str, torch.Tensor]:
    """
    Prépare le graphe unique pour l'entraînement GNN
    
    Args:
        vectorized_graph: Votre graphe unique vectorisé
        max_nodes: Limite de nœuds à traiter (pour tests)
        
    Returns:
        Batch data pour le modèle
    """
    
    logger.info(f"Préparation du graphe: {vectorized_graph.number_of_nodes()} nœuds, {vectorized_graph.number_of_edges()} arêtes")
    
    # Limiter le nombre de nœuds si spécifié
    nodes_to_process = list(vectorized_graph.nodes())
    if max_nodes and len(nodes_to_process) > max_nodes:
        nodes_to_process = nodes_to_process[:max_nodes]
        logger.info(f"Limitation à {max_nodes} nœuds pour le traitement")
    
    # Créer le sous-graphe
    if max_nodes and len(nodes_to_process) < vectorized_graph.number_of_nodes():
        subgraph = vectorized_graph.subgraph(nodes_to_process).copy()
    else:
        subgraph = vectorized_graph
    
    # Mapping des IDs
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(subgraph.nodes())}
    
    # Extraire embeddings des nœuds
    node_embeddings = []
    node_types = []
    node_labels = []
    node_packet_ids = []
    
    for node_id, attrs in subgraph.nodes(data=True):
        if 'embedding' in attrs:
            node_embeddings.append(torch.tensor(attrs['embedding'], dtype=torch.float32))
            node_types.append(attrs.get('node_type', 0))
            node_labels.append(attrs.get('label', ''))
            node_packet_ids.append(attrs.get('packet_id', 0))
        else:
            logger.warning(f"Nœud {node_id} sans embedding")
            node_embeddings.append(torch.zeros(64, dtype=torch.float32))
            node_types.append(0)
            node_labels.append('')
            node_packet_ids.append(0)
    
    # Extraire embeddings des arêtes
    edge_embeddings = []
    edge_indices = []
    edge_labels = []
    
    for u, v, key, attrs in subgraph.edges(data=True, keys=True):
        if 'embedding' in attrs:
            edge_embeddings.append(torch.tensor(attrs['embedding'], dtype=torch.float32))
            edge_indices.append([old_to_new[u], old_to_new[v]])
            edge_labels.append(attrs.get('label', ''))
        else:
            logger.warning(f"Arête {u}->{v} sans embedding")
            edge_embeddings.append(torch.zeros(64, dtype=torch.float32))
            edge_indices.append([old_to_new[u], old_to_new[v]])
            edge_labels.append('')
    
    # Identifier nœuds partagés
    shared_node_info = _identify_shared_nodes(node_labels, node_packet_ids, node_embeddings)
    
    # Créer timestamps basés sur packet_id
    unique_packet_ids = list(set(node_packet_ids))
    timestamps = torch.tensor([float(pid) for pid in unique_packet_ids], dtype=torch.float32)
    
    batch_data = {
        'node_embeddings': torch.stack(node_embeddings) if node_embeddings else torch.empty(0, 64),
        'edge_embeddings': torch.stack(edge_embeddings) if edge_embeddings else torch.empty(0, 64),
        'edge_index': torch.tensor(edge_indices, dtype=torch.long).t().contiguous() if edge_indices else torch.empty(2, 0, dtype=torch.long),
        'node_types': torch.tensor(node_types, dtype=torch.long) if node_types else torch.empty(0, dtype=torch.long),
        'timestamps': timestamps,
        'shared_node_info': shared_node_info if shared_node_info['node_ids'] else None
    }
    
    logger.info(f"Données préparées: {len(node_embeddings)} nœuds, {len(edge_embeddings)} arêtes, {len(shared_node_info['node_ids'])} nœuds partagés")
    
    return batch_data

def _identify_shared_nodes(node_labels: List[str], 
                         node_packet_ids: List[int],
                         node_embeddings: List[torch.Tensor]) -> Dict[str, Any]:
    """Identifie les nœuds partagés entre paquets"""
    
    label_to_packets = defaultdict(set)
    label_to_embedding = {}
    
    for i, (label, packet_id) in enumerate(zip(node_labels, node_packet_ids)):
        if label.strip() and _is_shareable_node(label):
            label_to_packets[label].add(packet_id)
            if label not in label_to_embedding:
                label_to_embedding[label] = node_embeddings[i]
    
    shared_node_info = {
        'node_ids': [],
        'embeddings': [],
        'packet_associations': []
    }
    
    for label, packet_set in label_to_packets.items():
        if len(packet_set) > 1:  # Partagé entre plusieurs paquets
            shared_node_info['node_ids'].append(label)
            shared_node_info['embeddings'].append(label_to_embedding[label])
            shared_node_info['packet_associations'].append(list(packet_set))
    
    if shared_node_info['embeddings']:
        shared_node_info['embeddings'] = torch.stack(shared_node_info['embeddings'])
    else:
        shared_node_info['embeddings'] = torch.empty(0, 64)
    
    logger.info(f"Trouvé {len(shared_node_info['node_ids'])} nœuds partagés")
    
    return shared_node_info

def _is_shareable_node(label: str) -> bool:
    """Détermine si un nœud peut être partagé"""
    
    # IP addresses
    if _looks_like_ip(label):
        return True
    
    # Services 5G
    if any(service in label.lower() for service in ['nudm', 'namf', 'nsmf', 'npcf', 'nausf', 'nnrf']):
        return True
    
    # UUIDs
    if len(label) == 36 and label.count('-') == 4:
        return True
    
    # Ports
    try:
        port = int(label)
        return 1 <= port <= 65535
    except ValueError:
        pass
    
    return False

def _looks_like_ip(value: str) -> bool:
    """Vérifie si c'est une IP"""
    parts = value.split('.')
    if len(parts) != 4:
        return False
    
    try:
        return all(0 <= int(part) <= 255 for part in parts)
    except ValueError:
        return False

def create_training_batches(batch_data: Dict[str, torch.Tensor], 
                          batch_size: int = 1000) -> List[Dict[str, torch.Tensor]]:
    """
    Crée des sous-batches depuis le graphe complet
    
    Args:
        batch_data: Données du graphe complet
        batch_size: Nombre de nœuds par sous-batch
        
    Returns:
        Liste de sous-batches
    """
    
    total_nodes = len(batch_data['node_embeddings'])
    
    if total_nodes <= batch_size:
        return [batch_data]
    
    batches = []
    
    for start_idx in range(0, total_nodes, batch_size):
        end_idx = min(start_idx + batch_size, total_nodes)
        
        # Nœuds du sous-batch
        batch_node_embeddings = batch_data['node_embeddings'][start_idx:end_idx]
        batch_node_types = batch_data['node_types'][start_idx:end_idx]
        
        # Arêtes qui connectent ces nœuds
        edge_mask = (batch_data['edge_index'][0] >= start_idx) & (batch_data['edge_index'][0] < end_idx) & \
                   (batch_data['edge_index'][1] >= start_idx) & (batch_data['edge_index'][1] < end_idx)
        
        batch_edge_indices = batch_data['edge_index'][:, edge_mask] - start_idx  # Ajuster les indices
        batch_edge_embeddings = batch_data['edge_embeddings'][edge_mask]
        
        # Nœuds partagés pour ce sous-batch (optionnel pour pré-entraînement)
        shared_info = None
        if batch_data['shared_node_info'] is not None:
            shared_info = {
                'node_ids': batch_data['shared_node_info']['node_ids'][:min(10, len(batch_data['shared_node_info']['node_ids']))],
                'embeddings': batch_data['shared_node_info']['embeddings'][:min(10, len(batch_data['shared_node_info']['embeddings']))],
                'packet_associations': batch_data['shared_node_info']['packet_associations'][:min(10, len(batch_data['shared_node_info']['packet_associations']))]
            }
        
        batch = {
            'node_embeddings': batch_node_embeddings,
            'edge_embeddings': batch_edge_embeddings,
            'edge_index': batch_edge_indices,
            'node_types': batch_node_types,
            'timestamps': batch_data['timestamps'],
            'shared_node_info': shared_info
        }
        
        batches.append(batch)
    
    logger.info(f"Créé {len(batches)} sous-batches de ~{batch_size} nœuds chacun")
    
    return batches

def validate_graph_data(vectorized_graph: nx.MultiDiGraph) -> Dict[str, Any]:
    """
    Valide les données du graphe
    
    Args:
        vectorized_graph: Graphe à valider
        
    Returns:
        Rapport de validation
    """
    
    validation_results = {
        'valid': True,
        'issues': [],
        'stats': {
            'total_nodes': vectorized_graph.number_of_nodes(),
            'total_edges': vectorized_graph.number_of_edges(),
            'node_types': set(),
            'entity_types': set(),
            'packet_ids': set(),
            'embedding_dims': set()
        }
    }
    
    logger.info(f"Validation du graphe: {validation_results['stats']['total_nodes']} nœuds, {validation_results['stats']['total_edges']} arêtes")
    
    # Valider les nœuds
    nodes_without_embedding = 0
    for node_id, attrs in vectorized_graph.nodes(data=True):
        
        if 'embedding' not in attrs:
            nodes_without_embedding += 1
        else:
            validation_results['stats']['embedding_dims'].add(len(attrs['embedding']))
        
        validation_results['stats']['node_types'].add(attrs.get('node_type', 'unknown'))
        validation_results['stats']['entity_types'].add(attrs.get('entity_type', 'unknown'))
        validation_results['stats']['packet_ids'].add(attrs.get('packet_id', 0))
    
    # Valider les arêtes
    edges_without_embedding = 0
    for u, v, key, attrs in vectorized_graph.edges(data=True, keys=True):
        if 'embedding' not in attrs:
            edges_without_embedding += 1
        else:
            validation_results['stats']['embedding_dims'].add(len(attrs['embedding']))
    
    # Vérifications
    if nodes_without_embedding > 0:
        validation_results['issues'].append(f"{nodes_without_embedding} nœuds sans embedding")
        if nodes_without_embedding > validation_results['stats']['total_nodes'] * 0.1:
            validation_results['valid'] = False
    
    if edges_without_embedding > 0:
        validation_results['issues'].append(f"{edges_without_embedding} arêtes sans embedding")
        if edges_without_embedding > validation_results['stats']['total_edges'] * 0.1:
            validation_results['valid'] = False
    
    if len(validation_results['stats']['embedding_dims']) > 1:
        validation_results['issues'].append(f"Dimensions d'embedding incohérentes: {validation_results['stats']['embedding_dims']}")
        validation_results['valid'] = False
    
    # Convertir sets en listes
    for key, value in validation_results['stats'].items():
        if isinstance(value, set):
            validation_results['stats'][key] = list(value)
    
    validation_results['stats']['unique_packets'] = len(validation_results['stats']['packet_ids'])
    
    logger.info(f"Validation: {' OK' if validation_results['valid'] else ' ERREUR'}")
    
    return validation_results