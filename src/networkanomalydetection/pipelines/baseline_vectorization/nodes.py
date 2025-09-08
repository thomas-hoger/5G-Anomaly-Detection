""" Nœuds du pipeline de vectorisation baseline """
import networkx as nx
import logging
import pickle
from typing import Dict, Any, Tuple

from networkanomalydetection.core.feature_vectorization.baseline_vectorizer import BaselineVectorizer

logger = logging.getLogger(__name__)

def baseline_vectorize_graph_node(complete_graph_pickle: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Vectorisation baseline avec GMM + TF-IDF - retourne SEULEMENT le graphe
    
    Args:
        complete_graph_pickle: Graphe depuis graph_construction
        
    Returns:
        nx.MultiDiGraph: Graphe vectorisé
    """
    logger.info("Début vectorisation baseline")
    
    # DEBUG: Vérifier le type de l'input
    logger.info(f"Type de l'input reçu: {type(complete_graph_pickle)}")
    
    # Si c'est une string, essayer de la désérialiser
    if isinstance(complete_graph_pickle, str):
        logger.warning("Input est une string, tentative de désérialisation...")
        try:
            # Si c'est du JSON
            import json
            graph_data = json.loads(complete_graph_pickle)
            complete_graph_pickle = nx.node_link_graph(graph_data)
            logger.info("Désérialisation JSON réussie")
        except:
            try:
                # Si c'est du pickle en string
                complete_graph_pickle = pickle.loads(complete_graph_pickle.encode())
                logger.info("Désérialisation pickle réussie")
            except Exception as e:
                logger.error(f"Impossible de désérialiser: {e}")
                raise ValueError(f"Input invalide - Type: {type(complete_graph_pickle)}")
    
    # Vérifier que c'est bien un graphe NetworkX
    if not isinstance(complete_graph_pickle, nx.MultiDiGraph):
        logger.error(f"Type attendu: nx.MultiDiGraph, reçu: {type(complete_graph_pickle)}")
        raise TypeError(f"Input doit être un nx.MultiDiGraph, reçu: {type(complete_graph_pickle)}")
    
    logger.info(f"Graphe valide - Nœuds: {complete_graph_pickle.number_of_nodes()}, Arêtes: {complete_graph_pickle.number_of_edges()}")
    
    vectorizer = BaselineVectorizer(target_dim=64)
    vectorized_graph, _ = vectorizer.vectorize_graph(complete_graph_pickle)
    
    logger.info(f"Type du graphe vectorisé: {type(vectorized_graph)}")
    logger.info(f"Nombre de nœuds: {vectorized_graph.number_of_nodes()}")
    logger.info("Vectorisation baseline terminée")
    
    return vectorized_graph

def baseline_generate_report_node(complete_graph_pickle: nx.MultiDiGraph) -> str:
    """
    Génère le rapport de vectorisation baseline
    
    Args:
        complete_graph_pickle: Graphe depuis graph_construction
        
    Returns:
        str: Rapport JSON
    """
    logger.info("Génération rapport vectorisation baseline")
    
    if isinstance(complete_graph_pickle, str):
        logger.warning("Input rapport est une string, tentative de désérialisation...")
        try:
            import json
            graph_data = json.loads(complete_graph_pickle)
            complete_graph_pickle = nx.node_link_graph(graph_data)
        except:
            try:
                complete_graph_pickle = pickle.loads(complete_graph_pickle.encode())
            except Exception as e:
                logger.error(f"Impossible de désérialiser pour rapport: {e}")
                return json.dumps({"error": f"Input invalide: {type(complete_graph_pickle)}"})
    
    if not isinstance(complete_graph_pickle, nx.MultiDiGraph):
        import json
        return json.dumps({"error": f"Type attendu: nx.MultiDiGraph, reçu: {type(complete_graph_pickle)}"})
    
    vectorizer = BaselineVectorizer(target_dim=64)
    _, report_json = vectorizer.vectorize_graph(complete_graph_pickle)
    
    return report_json