"""
Utilitaires pour la préparation des données GNN
"""
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


def convert_dict_to_pytorch_geometric(subgraph_dict: Dict) -> Data:
    # Debug: Check what keys are available
    print(f"Subgraph dict keys: {subgraph_dict.keys()}")
    
    # Debug: Check if node_type exists
    if 'node_type' in subgraph_dict:
        print(f"Node types found: {subgraph_dict['node_type']}")
    else:
        print("WARNING: No node_type in subgraph_dict!")
    
    return Data(
        x=subgraph_dict['x'],
        edge_index=subgraph_dict['edge_index'],
        edge_attr=subgraph_dict['edge_attr'],
        num_nodes=subgraph_dict.get('num_nodes'),
        synthetic_edge=subgraph_dict.get('synthetic_edge', False),
        node_type=subgraph_dict.get('node_type')
    )


def extract_packet_ids(data_list: List[Dict]) -> np.ndarray:
    """Extraire les packet_ids des données"""
    packet_ids = []
    for batch in data_list:
        packet_ids.append(batch['packet_id'])
    return np.array(packet_ids)


def temporal_train_val_split(data_list: List[Dict], 
                           train_ratio: float = 0.7,
                           val_ratio: float = 0.3) -> Tuple[List, List]:
    """
    Split temporel des données basé sur les packet_ids
    
    Args:
        data_list: Liste de {'packet_id': int, 'subgraphs': list, 'count': int}
        train_ratio: Proportion pour l'entraînement
        val_ratio: Proportion pour la validation
    
    Returns:
        train_data, val_data
    """
    # Extraire packet_ids pour le tri temporel
    packet_ids = extract_packet_ids(data_list)
    sorted_indices = np.argsort(packet_ids)
    
    # Split temporel
    n_total = len(data_list)
    n_train = int(n_total * train_ratio)
    
    train_indices = sorted_indices[:n_train]
    val_indices = sorted_indices[n_train:]
    
    train_data = [data_list[i] for i in train_indices]
    val_data = [data_list[i] for i in val_indices]
    
    logger.info(f"Split temporel: {len(train_data)} train, {len(val_data)} val")
    logger.info(f"Train packet_ids: {packet_ids[train_indices].min()}-{packet_ids[train_indices].max()}")
    logger.info(f"Val packet_ids: {packet_ids[val_indices].min()}-{packet_ids[val_indices].max()}")
    
    return train_data, val_data


def flatten_subgraphs(data_list: List[Dict]) -> List[Data]:
    """
    Aplatir la liste de {'packet_id': int, 'subgraphs': list} en liste de subgraphs PyTorch Geometric
    """
    flattened = []
    for batch in data_list:
        for subgraph_dict in batch['subgraphs']:
            # Convertir dict → PyTorch Geometric Data
            subgraph = convert_dict_to_pytorch_geometric(subgraph_dict)
            flattened.append(subgraph)
    
    logger.info(f"Aplatissement: {len(data_list)} batches → {len(flattened)} subgraphs PyTorch Geometric")
    return flattened


def create_data_loaders(train_data: List[Dict], 
                       val_data: List[Dict],
                       batch_size: int = 32,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Créer les DataLoaders PyTorch Geometric
    
    Args:
        train_data, val_data: Données splitées (format dict)
        batch_size: Taille des batches
        num_workers: Nombre de workers
    
    Returns:
        train_loader, val_loader
    """
    # Aplatir les données (avec conversion automatique)
    train_subgraphs = flatten_subgraphs(train_data)
    val_subgraphs = flatten_subgraphs(val_data)
    
    # Créer les DataLoaders
    train_loader = DataLoader(
        train_subgraphs,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subgraphs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"DataLoaders créés: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    return train_loader, val_loader


def compute_anomaly_thresholds(model, val_loader, device, percentile: float = 95):
    """
     Calculer les seuils d'anomalie tri-domaine sur les données de validation
    
    Args:
        model: Modèle entraîné
        val_loader: DataLoader de validation
        device: Device PyTorch
        percentile: Percentile pour le seuil
    
    Returns:
        dict avec seuils par domaine tri-domaine
    """
    model.eval()
    
    #  Séparer les erreurs par domaine
    central_errors = []
    param_errors = []
    edge_errors = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs = model(batch)
            scores = model.compute_anomaly_scores(outputs, batch)
            
            #  Utiliser les bonnes clés tri-domaine
            if len(scores['central_anomaly_scores']) > 0:
                central_errors.extend(scores['central_anomaly_scores'].cpu().numpy())
            
            if len(scores['param_anomaly_scores']) > 0:
                param_errors.extend(scores['param_anomaly_scores'].cpu().numpy())
            
            edge_errors.extend(scores['edge_anomaly_scores'].cpu().numpy())
    
    #  Calculer seuils tri-domaine
    thresholds = {}
    
    if central_errors:
        thresholds['central_threshold'] = np.percentile(central_errors, percentile)
        logger.info(f"Seuil nœuds centraux ({percentile}ème percentile): {thresholds['central_threshold']:.4f}")
    
    if param_errors:
        thresholds['param_threshold'] = np.percentile(param_errors, percentile)
        logger.info(f"Seuil nœuds paramètres ({percentile}ème percentile): {thresholds['param_threshold']:.4f}")
    
    if edge_errors:
        thresholds['edge_threshold'] = np.percentile(edge_errors, percentile)
        logger.info(f"Seuil edges ({percentile}ème percentile): {thresholds['edge_threshold']:.4f}")
    
    return thresholds

def analyze_data_statistics(data_list: List[Dict]) -> Dict[str, Any]:
    """
    Analyser les statistiques des données (après conversion PyTorch Geometric)
    """
    subgraphs = flatten_subgraphs(data_list)  # Maintenant ce sont des Data objects
    
    num_nodes = [data.x.size(0) for data in subgraphs]
    num_edges = [data.edge_index.size(1) for data in subgraphs]
    
    stats = {
        'total_subgraphs': len(subgraphs),
        'nodes_min': min(num_nodes),
        'nodes_max': max(num_nodes),
        'nodes_mean': np.mean(num_nodes),
        'edges_min': min(num_edges),
        'edges_max': max(num_edges),
        'edges_mean': np.mean(num_edges),
    }
    
    logger.info(f"Statistiques données:")
    logger.info(f"  Subgraphs: {stats['total_subgraphs']}")
    logger.info(f"  Nœuds: {stats['nodes_min']}-{stats['nodes_max']} (moy: {stats['nodes_mean']:.1f})")
    logger.info(f"  Edges: {stats['edges_min']}-{stats['edges_max']} (moy: {stats['edges_mean']:.1f})")
    
    return stats