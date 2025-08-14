"""
This is a boilerplate pipeline 'baseline_vectorization'
generated using Kedro 1.0.0
"""
"""
Nœuds du pipeline de vectorisation baseline
"""
"""
Nœuds du pipeline de vectorisation baseline
"""
import networkx as nx
import logging
from typing import Dict, Any, Tuple

from networkanomalydetection.core.baseline_vectorization.baseline_vectorizer import BaselineVectorizer

logger = logging.getLogger(__name__)

def baseline_vectorize_graph_node(complete_graph_pickle: nx.MultiDiGraph) -> Tuple[nx.MultiDiGraph, str]:
    """
    Vectorisation baseline avec GMM + TF-IDF
    
    Args:
        complete_graph_pickle: Graphe depuis graph_construction
        
    Returns:
        Tuple[nx.MultiDiGraph, str]: (Graphe vectorisé, Rapport JSON)
    """
    logger.info("Début vectorisation baseline")
    
    vectorizer = BaselineVectorizer(target_dim=64)
    vectorized_graph, report_json = vectorizer.vectorize_graph(complete_graph_pickle)
    
    logger.info("Vectorisation baseline terminée")
    return vectorized_graph, report_json