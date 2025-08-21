"""
Nodes pipeline GNN Sampling
"""
import logging
from typing import Dict, List, Any
import networkx as nx

logger = logging.getLogger(__name__)

def generate_subgraphs_main(vectorized_graph: nx.MultiDiGraph) -> Dict[str, Any]:
    """Générer subgraphs pour vectorisation principale"""
    
    from networkanomalydetection.core.gnn.sampling.subgraph_generator import create_training_pipeline
    
    logger.info(f"Génération subgraphs main: {vectorized_graph.number_of_nodes()} noeuds")
    
    # Pipeline de génération
    pipeline = create_training_pipeline(vectorized_graph)
    
    # Collecter tous les batches
    all_batches = []
    total_subgraphs = 0
    
    for packet_id, pytorch_subgraphs in pipeline:
        batch_data = {
            'packet_id': packet_id,
            'subgraphs': pytorch_subgraphs,
            'count': len(pytorch_subgraphs)
        }
        all_batches.append(batch_data)
        total_subgraphs += len(pytorch_subgraphs)
        
        if len(all_batches) % 50 == 0:
            logger.info(f"Traité {len(all_batches)} batches")
    
    result = {
        'batches': all_batches,
        'total_batches': len(all_batches),
        'total_subgraphs': total_subgraphs
    }
    
    logger.info(f"Génération terminée: {len(all_batches)} batches, {total_subgraphs} subgraphs")
    return result

def generate_subgraphs_baseline(baseline_vectorized_graph: nx.MultiDiGraph) -> Dict[str, Any]:
    """Générer subgraphs pour vectorisation baseline"""
    
    from networkanomalydetection.core.gnn.sampling.subgraph_generator import create_training_pipeline
    
    logger.info(f"Génération subgraphs baseline: {baseline_vectorized_graph.number_of_nodes()} noeuds")
    
    # Pipeline de génération
    pipeline = create_training_pipeline(baseline_vectorized_graph)
    
    # Collecter tous les batches
    all_batches = []
    total_subgraphs = 0
    
    for packet_id, pytorch_subgraphs in pipeline:
        batch_data = {
            'packet_id': packet_id,
            'subgraphs': pytorch_subgraphs,
            'count': len(pytorch_subgraphs)
        }
        all_batches.append(batch_data)
        total_subgraphs += len(pytorch_subgraphs)
        
        if len(all_batches) % 50 == 0:
            logger.info(f"Baseline - Traité {len(all_batches)} batches")
    
    result = {
        'batches': all_batches,
        'total_batches': len(all_batches),
        'total_subgraphs': total_subgraphs
    }
    
    logger.info(f"Génération baseline terminée: {len(all_batches)} batches, {total_subgraphs} subgraphs")
    return result

def prepare_training_data_main(subgraphs_main: Dict[str, Any]) -> Dict[str, Any]:
    """Préparer données finales pour entraînement principal"""
    
    logger.info("Préparation données main")
    
    batches = subgraphs_main['batches']
    
    # Validation simple
    if not batches:
        logger.error("Aucun batch généré")
        return {'status': 'ERROR', 'batches': []}
    
    # Statistiques
    subgraph_counts = [b['count'] for b in batches]
    avg_subgraphs = sum(subgraph_counts) / len(subgraph_counts)
    
    prepared_data = {
        'batches': batches,
        'stats': {
            'total_batches': len(batches),
            'total_subgraphs': subgraphs_main['total_subgraphs'],
            'avg_subgraphs_per_batch': avg_subgraphs
        },
        'status': 'READY'
    }
    
    logger.info(f"Données main prêtes: {len(batches)} batches")
    return prepared_data

def prepare_training_data_baseline(subgraphs_baseline: Dict[str, Any]) -> Dict[str, Any]:
    """Préparer données finales pour entraînement baseline"""
    
    logger.info("Préparation données baseline")
    
    batches = subgraphs_baseline['batches']
    
    # Validation simple
    if not batches:
        logger.error("Aucun batch baseline généré")
        return {'status': 'ERROR', 'batches': []}
    
    # Statistiques
    subgraph_counts = [b['count'] for b in batches]
    avg_subgraphs = sum(subgraph_counts) / len(subgraph_counts)
    
    prepared_data = {
        'batches': batches,
        'stats': {
            'total_batches': len(batches),
            'total_subgraphs': subgraphs_baseline['total_subgraphs'],
            'avg_subgraphs_per_batch': avg_subgraphs
        },
        'status': 'READY'
    }
    
    logger.info(f"Données baseline prêtes: {len(batches)} batches")
    return prepared_data