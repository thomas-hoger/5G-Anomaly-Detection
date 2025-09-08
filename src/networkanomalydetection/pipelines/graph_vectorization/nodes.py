"""
This is a boilerplate pipeline 'gnn_conversion'
generated using Kedro 1.0.0
"""
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

def convert_networkx_to_pytorch(vectorized_graph) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Convertit un graphe NetworkX vectorisé au format PyTorch Geometric
    
    Args:
        vectorized_graph: Graphe NetworkX avec embeddings
        
    Returns:
        Tuple (pytorch_data, conversion_metadata, validation_report)
    """
    from networkanomalydetection.core.graph_vectorization import NetworkXToTorchConverter
    
    logger.info(f"Conversion du graphe NetworkX: {vectorized_graph.number_of_nodes()} nœuds")
    
    # Configuration du convertisseur
    converter = NetworkXToTorchConverter(
        validate=True,      # Activer la validation
        device='cpu'        # Utiliser CPU par défaut
    )
    
    # Conversion
    pytorch_data, metadata, validation_report = converter.convert(vectorized_graph)
    
    # Log des résultats
    logger.info(f"Conversion réussie: {validation_report.get('overall_status', 'UNKNOWN')}")
    logger.info(f"Données PyTorch: x={pytorch_data['x'].shape}, edge_index={pytorch_data['edge_index'].shape}")
    
    return pytorch_data, metadata, validation_report

def validate_conversion_quality(validation_report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyse la qualité de la conversion et génère un rapport
    
    Args:
        validation_report: Rapport de validation de la conversion
        
    Returns:
        Rapport de qualité enrichi
    """
    logger.info("Analyse de la qualité de conversion")
    
    # Calcul du score de qualité global
    tests_passed = validation_report.get('tests_passed', 0)
    tests_failed = validation_report.get('tests_failed', 0)
    total_tests = tests_passed + tests_failed
    
    if total_tests > 0:
        quality_score = tests_passed / total_tests
    else:
        quality_score = 0.0
    
    # Classification de la qualité
    if quality_score >= 0.9:
        quality_level = "EXCELLENT"
    elif quality_score >= 0.75:
        quality_level = "GOOD"
    elif quality_score >= 0.5:
        quality_level = "ACCEPTABLE"
    else:
        quality_level = "POOR"
    
    quality_report = {
        'quality_score': quality_score,
        'quality_level': quality_level,
        'tests_summary': {
            'passed': tests_passed,
            'failed': tests_failed,
            'total': total_tests
        },
        'recommendations': [],
        'warnings_count': len(validation_report.get('warnings', [])),
        'errors_count': len(validation_report.get('errors', []))
    }
    
    # Recommandations basées sur les résultats
    if quality_score < 0.9:
        quality_report['recommendations'].append("Vérifier les tests échoués pour améliorer la qualité")
    
    if validation_report.get('errors'):
        quality_report['recommendations'].append("Résoudre les erreurs critiques identifiées")
    
    if validation_report.get('warnings'):
        quality_report['recommendations'].append("Examiner les avertissements pour optimisations possibles")
    
    logger.info(f"Score de qualité: {quality_score:.3f} ({quality_level})")
    return quality_report