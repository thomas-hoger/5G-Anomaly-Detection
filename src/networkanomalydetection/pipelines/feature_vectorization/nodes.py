"""
This is a boilerplate pipeline 'vectorization'
generated using Kedro 1.0.0rc1
"""
"""
Nœuds Kedro pour le pipeline de vectorisation
"""
import networkx as nx
import logging
from typing import Dict, Any, Tuple,List
import datetime


from networkanomalydetection.core.feature_vectorization.graph_vectorizer import IntelligentGraphVectorizer

logger = logging.getLogger(__name__)

def vectorize_graph_node(complete_graph_pickle: nx.MultiDiGraph, 
                        parameters: Dict[str, Any] = None) -> Tuple[nx.MultiDiGraph, Dict[str, Any]]:
    """
    Nœud principal de vectorisation du graphe
    
    Args:
        complete_graph_pickle: Graphe NetworkX chargé automatiquement par Kedro depuis le pickle
        parameters: Paramètres de configuration depuis parameters.yml
        
    Returns:
        Tuple[nx.MultiDiGraph, Dict]: Graphe vectorisé et rapport de vectorisation
    """
    
    # Paramètres par défaut (mergés avec ceux du fichier parameters.yml)
    config = {
        'node_dim': 64,
        'edge_dim': 64,
        'log_level': 'INFO'
    }
    
    # Merger avec les paramètres Kedro
    if parameters:
        config.update(parameters)
    
    logger.info(f" Starting graph vectorization with config: {config}")
    logger.info(f" Input graph: {complete_graph_pickle.number_of_nodes()} nodes, {complete_graph_pickle.number_of_edges()} edges")
    
    try:
        # Créer le vectoriseur intelligent
        vectorizer = IntelligentGraphVectorizer(
            node_dim=config['node_dim'],
            edge_dim=config['edge_dim']
        )
        
        logger.info(" Fitting vectorizer on graph data...")
        
        # Vectoriser le graphe (fit + transform automatique)
        vectorized_graph, report = vectorizer.fit_transform(complete_graph_pickle)
        
        # Créer le rapport de vectorisation pour Kedro
        vectorization_report = vectorizer.get_vectorization_summary(report)
        
        # Ajouter timestamp et métadonnées Kedro
        vectorization_report['pipeline_metadata'] = {
            'timestamp': datetime.datetime.now().isoformat(),
            'kedro_pipeline': 'vectorization',
            'node_name': 'vectorize_graph_node',
            'input_graph_nodes': complete_graph_pickle.number_of_nodes(),
            'input_graph_edges': complete_graph_pickle.number_of_edges(),
            'config_used': config
        }
        
        # Logs de résultats pour Kedro
        logger.info(" Vectorization completed successfully!")
        logger.info(f"  Node classifications: {vectorization_report['node_classifications']}")
        logger.info(f" Edge classifications: {vectorization_report['edge_classifications']}")
        logger.info(f"  Encoding methods used: {vectorization_report['encoding_methods']}")
        logger.info(f" Overall success rate: {vectorization_report['quality_metrics']['overall_success_rate']:.1f}%")
        
        # Warning si erreurs
        if report.errors:
            logger.warning(f"  Vectorization completed with {len(report.errors)} errors")
            for error in report.errors[:3]:  # Log first 3 errors only
                logger.warning(f"   - {error}")
        
        # Validation finale du graphe de sortie
        _validate_output_graph(vectorized_graph, config)
        
        logger.info(" Graph vectorization node completed successfully")
        
        return vectorized_graph, vectorization_report
        
    except Exception as e:
        logger.error(f" Error during graph vectorization: {e}")
        # Re-raise pour que Kedro puisse gérer l'erreur
        raise

def create_vectorization_summary_node(vectorization_report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Crée un résumé détaillé de la vectorisation pour dashboard/reporting
    
    Args:
        vectorization_report: Rapport de vectorisation depuis le nœud précédent
        
    Returns:
        Dict: Résumé formaté pour sauvegarde et monitoring
    """
    
    logger.info(" Creating detailed vectorization summary...")
    
    try:
        # Extraire les métriques clés
        total_nodes = vectorization_report.get('total_nodes', 0)
        total_edges = vectorization_report.get('total_edges', 0)
        
        node_classifications = vectorization_report.get('node_classifications', {})
        edge_classifications = vectorization_report.get('edge_classifications', {})
        quality_metrics = vectorization_report.get('quality_metrics', {})
        
        # Calculer des statistiques additionnelles
        most_common_node_type = max(node_classifications.items(), key=lambda x: x[1]) if node_classifications else ('UNKNOWN', 0)
        most_common_edge_type = max(edge_classifications.items(), key=lambda x: x[1]) if edge_classifications else ('UNKNOWN', 0)
        
        # Taux de succès détaillés
        overall_success_rate = quality_metrics.get('overall_success_rate', 0)
        node_success_rate = quality_metrics.get('node_success_rate', 0)
        edge_success_rate = quality_metrics.get('edge_success_rate', 0)
        
        # Créer le résumé structuré
        summary = {
            'vectorization_summary': {
                'execution_info': {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'pipeline': 'vectorization',
                    'status': 'SUCCESS' if overall_success_rate > 95 else 'WARNING' if overall_success_rate > 80 else 'ERROR'
                },
                'graph_metrics': {
                    'total_entities': total_nodes + total_edges,
                    'nodes_count': total_nodes,
                    'edges_count': total_edges,
                    'graph_density': total_edges / max(total_nodes * (total_nodes - 1), 1) if total_nodes > 1 else 0
                },
                'classification_results': {
                    'nodes': {
                        'classifications': node_classifications,
                        'most_common_type': most_common_node_type[0],
                        'most_common_count': most_common_node_type[1],
                        'unique_types': len(node_classifications)
                    },
                    'edges': {
                        'classifications': edge_classifications,
                        'most_common_type': most_common_edge_type[0],
                        'most_common_count': most_common_edge_type[1],
                        'unique_types': len(edge_classifications)
                    }
                },
                'quality_assessment': {
                    'overall_success_rate_percent': round(overall_success_rate, 2),
                    'node_success_rate_percent': round(node_success_rate, 2),
                    'edge_success_rate_percent': round(edge_success_rate, 2),
                    'errors_count': vectorization_report.get('errors_count', 0),
                    'quality_grade': _calculate_quality_grade(overall_success_rate)
                },
                'technical_details': {
                    'encoding_methods': vectorization_report.get('encoding_methods', {}),
                    'dimensions': vectorization_report.get('dimensions', {}),
                    'graph_statistics': {
                        'node_stats': vectorization_report.get('node_stats', {}),
                        'edge_stats': vectorization_report.get('edge_stats', {})
                    }
                },
                'recommendations': _generate_recommendations(vectorization_report)
            }
        }
        
        # Logs de confirmation
        logger.info(f" Summary created for {total_nodes + total_edges} total entities")
        logger.info(f" Quality grade: {summary['vectorization_summary']['quality_assessment']['quality_grade']}")
        logger.info(f" Success rate: {overall_success_rate:.1f}%")
        
        return summary
        
    except Exception as e:
        logger.error(f" Error creating vectorization summary: {e}")
        # Retourner un résumé minimal en cas d'erreur
        return {
            'vectorization_summary': {
                'execution_info': {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'status': 'ERROR',
                    'error': str(e)
                }
            }
        }

def _validate_output_graph(graph: nx.MultiDiGraph, config: Dict[str, Any]):
    """Valide le graphe de sortie pour Kedro"""
    
    logger.info("🔍 Validating output graph...")
    
    # Vérifications de base
    assert graph.number_of_nodes() > 0, "Output graph has no nodes"
    assert graph.number_of_edges() > 0, "Output graph has no edges"
    
    # Vérifier les embeddings des nœuds
    nodes_with_embeddings = 0
    for node_id, attrs in graph.nodes(data=True):
        if 'embedding' in attrs:
            nodes_with_embeddings += 1
            embedding = attrs['embedding']
            assert len(embedding) == config['node_dim'], \
                f"Node {node_id} embedding has wrong dimension: {len(embedding)} != {config['node_dim']}"
            assert 'entity_type' in attrs, f"Node {node_id} missing entity_type"
            assert 'classification_confidence' in attrs, f"Node {node_id} missing confidence"
    
    assert nodes_with_embeddings == graph.number_of_nodes(), \
        f"Only {nodes_with_embeddings}/{graph.number_of_nodes()} nodes have embeddings"
    
    # Vérifier les embeddings des arêtes
    edges_with_embeddings = 0
    for u, v, attrs in graph.edges(data=True):
        if 'embedding' in attrs:
            edges_with_embeddings += 1
            embedding = attrs['embedding']
            assert len(embedding) == config['edge_dim'], \
                f"Edge {u}->{v} embedding has wrong dimension: {len(embedding)} != {config['edge_dim']}"
            assert 'entity_type' in attrs, f"Edge {u}->{v} missing entity_type"
            assert 'classification_confidence' in attrs, f"Edge {u}->{v} missing confidence"
    
    assert edges_with_embeddings == graph.number_of_edges(), \
        f"Only {edges_with_embeddings}/{graph.number_of_edges()} edges have embeddings"
    
    logger.info(" Output graph validation passed successfully")

def _calculate_quality_grade(success_rate: float) -> str:
    """Calcule une note de qualité basée sur le taux de succès"""
    if success_rate >= 98:
        return "EXCELLENT"
    elif success_rate >= 95:
        return "VERY_GOOD"
    elif success_rate >= 90:
        return "GOOD"
    elif success_rate >= 80:
        return "FAIR"
    else:
        return "POOR"

def _generate_recommendations(report: Dict[str, Any]) -> List[str]:
    """Génère des recommandations basées sur le rapport"""
    recommendations = []
    
    success_rate = report.get('quality_metrics', {}).get('overall_success_rate', 0)
    errors_count = report.get('errors_count', 0)
    
    if success_rate < 95:
        recommendations.append("Consider increasing encoder confidence thresholds")
    
    if errors_count > 100:
        recommendations.append("High error count detected - review input data quality")
    
    node_types = report.get('node_classifications', {})
    if node_types.get('TEXT', 0) > node_types.get('SERVICE_5G', 0):
        recommendations.append("Many generic text nodes - consider adding specialized encoders")
    
    if not recommendations:
        recommendations.append("Vectorization quality is excellent - no recommendations")
    
    return recommendations