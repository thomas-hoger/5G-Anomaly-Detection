"""
This is a boilerplate pipeline 'graph_construction'
generated using Kedro 1.0.0rc1
"""
"""
Nœuds pour la construction du graphe topologique à partir du JSON complet
"""

import logging
import json
from typing import Dict, List, Any
from pathlib import Path

from ...utils.graph_topology import NetworkTopologyGraph

logger = logging.getLogger(__name__)

def load_dissected_packets_from_json(json_filepath: str) -> List[Dict[str, Any]]:
    """
    Charge tous les paquets dissectés depuis le fichier JSON
    
    Args:
        json_filepath: Chemin vers le fichier JSON des paquets dissectés
        
    Returns:
        List[Dict]: Liste complète des paquets dissectés
    """
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            dissected_packets = json.load(f)
        
        logger.info(f"Chargé {len(dissected_packets)} paquets depuis {json_filepath}")
        return dissected_packets
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement de {json_filepath}: {e}")
        return []

def build_complete_topology_graph(dissected_packets: List[Dict[str, Any]], 
                                 parameters: Dict[str, Any]) -> NetworkTopologyGraph:
    """
    Construit le graphe topologique complet à partir de TOUS les paquets dissectés
    
    Args:
        dissected_packets: Liste COMPLÈTE des paquets dissectés du JSON
        parameters: Paramètres de configuration
        
    Returns:
        NetworkTopologyGraph: Le graphe topologique complet
    """
    
    # Configuration
    graph_params = parameters.get("graph_construction", {})
    default_label = graph_params.get("default_central_node_label", "NETWORK_PACKET")
    batch_size = graph_params.get("batch_processing_size", 1000)
    
    # Créer le graphe
    graph_builder = NetworkTopologyGraph()
    
    total_packets = len(dissected_packets)
    logger.info(f"🚀 Construction du graphe topologique COMPLET avec {total_packets} paquets")
    logger.info(f"📊 Traitement par batch de {batch_size} paquets")
    
    processed_packets = 0
    failed_packets = 0
    
    # Traitement par batch pour éviter la surcharge mémoire
    for i in range(0, total_packets, batch_size):
        batch = dissected_packets[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_packets + batch_size - 1) // batch_size
        
        logger.info(f"📦 Traitement du batch {batch_num}/{total_batches} ({len(batch)} paquets)")
        
        for packet in batch:
            try:
                # Créer une copie pour éviter de modifier l'original
                packet_copy = packet.copy()
                
                # Déterminer le label du nœud central basé sur le contenu
                central_label = determine_central_node_label(packet_copy, default_label)
                
                # Ajouter le paquet au graphe
                node_id = graph_builder.add_packet(central_label, packet_copy)
                processed_packets += 1
                
            except Exception as e:
                failed_packets += 1
                logger.warning(f" Erreur paquet {processed_packets + failed_packets}: {e}")
                continue
        
        # Statistiques intermédiaires
        stats = graph_builder.get_graph_stats()
        logger.info(f"📈 Batch {batch_num} terminé - Nœuds: {stats['num_nodes']}, Arêtes: {stats['num_edges']}")
    
    # Statistiques finales
    stats = graph_builder.get_graph_stats()
    success_rate = (processed_packets / total_packets) * 100 if total_packets > 0 else 0
    
    logger.info(f" Graphe topologique COMPLET construit:")
    logger.info(f"   {processed_packets}/{total_packets} paquets traités ({success_rate:.1f}%)")
    logger.info(f"   {failed_packets} paquets échoués")
    logger.info(f"   {stats['num_nodes']} nœuds créés")
    logger.info(f"   {stats['num_edges']} arêtes créées")
    logger.info(f"   {stats['central_nodes']} nœuds centraux (paquets)")
    logger.info(f"   {stats['leaf_nodes']} nœuds feuilles (données)")
    
    return graph_builder

def determine_central_node_label(packet: Dict[str, Any], default_label: str) -> str:
    """
    Détermine le label du nœud central basé sur le contenu du paquet
    
    Args:
        packet: Le paquet dissecté
        default_label: Label par défaut
        
    Returns:
        str: Label approprié pour le nœud central
    """
    
    # Analyse du contenu HTTP2
    if "http2" in packet:
        http2_data = packet["http2"]
        
        # Requête HTTP avec méthode
        if "method" in http2_data:
            method = http2_data["method"]
            if "path" in http2_data:
                return f"HTTP2_{method}_{http2_data['path'].split('/')[-1]}"
            return f"HTTP2_{method}"
        
        # Réponse HTTP avec status
        elif "status" in http2_data:
            return f"HTTP2_RESPONSE_{http2_data['status']}"
        
        # JWT présent
        elif "jwt" in http2_data:
            return "HTTP2_JWT_AUTH"
        
        # Données utilisateur
        elif "ue_id" in http2_data:
            return f"HTTP2_UE_{http2_data['ue_id'][:8]}"  # Tronquer pour lisibilité
        
        # Générique HTTP2
        else:
            return "HTTP2_DATA"
    
    # Analyse des données communes
    common = packet.get("common", {})
    if common.get("is_request"):
        return "NETWORK_REQUEST"
    else:
        return "NETWORK_RESPONSE"

def extract_comprehensive_sequences(topology_graph: NetworkTopologyGraph,
                                  parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrait des séquences temporelles complètes du graphe entier
    
    Args:
        topology_graph: Le graphe topologique complet
        parameters: Paramètres de configuration
        
    Returns:
        Dict contenant toutes les séquences extraites du graphe complet
    """
    
    sequence_params = parameters.get("sequence_extraction", {})
    seq_length = sequence_params.get("sequence_length", 100)  # Plus long pour graphe complet
    max_ue_sequences = sequence_params.get("max_ue_sequences", 20)  # Limiter pour performance
    
    logger.info(f"🔍 Extraction de séquences du graphe complet (longueur: {seq_length})")
    
    sequences = {}
    
    # 1. Séquence générale (sans UE_id)
    general_seq = topology_graph.get_sequence(seq_len=seq_length, general=True)
    sequences["general"] = {
        "sequence": general_seq,
        "length": len(general_seq),
        "description": "Messages système sans ID utilisateur"
    }
    logger.info(f"📋 Séquence générale: {len(general_seq)} éléments")
    
    # 2. Séquence complète (tous les paquets)
    full_seq = topology_graph.get_sequence(seq_len=seq_length * 2)  # Plus longue
    sequences["complete"] = {
        "sequence": full_seq,
        "length": len(full_seq),
        "description": "Séquence temporelle complète de tous les paquets"
    }
    logger.info(f"📊 Séquence complète: {len(full_seq)} éléments")
    
    # 3. Séquences par UE_id (limitées pour performance)
    ue_ids = set()
    for node, data in topology_graph.get_graph().nodes(data=True):
        if "ue_id" in data:
            ue_ids.add(data["ue_id"])
    
    logger.info(f"👥 {len(ue_ids)} utilisateurs uniques détectés")
    
    sequences["by_user"] = {}
    processed_ues = 0
    
    for ue_id in sorted(ue_ids):  # Tri pour reproductibilité
        if processed_ues >= max_ue_sequences:
            logger.info(f"⏸️ Limite de {max_ue_sequences} utilisateurs atteinte")
            break
            
        ue_seq = topology_graph.get_sequence(seq_len=seq_length, specific_ue_id=ue_id)
        if len(ue_seq) > 5:  # Seulement si suffisamment de données
            sequences["by_user"][ue_id] = {
                "sequence": ue_seq,
                "length": len(ue_seq),
                "description": f"Séquence pour l'utilisateur {ue_id}"
            }
            processed_ues += 1
    
    logger.info(f"👤 Séquences utilisateurs: {len(sequences['by_user'])} utilisateurs analysés")
    
    # 4. Statistiques temporelles
    if full_seq:
        timestamps = [ts for _, ts in full_seq]
        sequences["temporal_stats"] = {
            "start_time": min(timestamps),
            "end_time": max(timestamps),
            "duration_seconds": max(timestamps) - min(timestamps),
            "total_packets": len(full_seq)
        }
    
    return sequences

def generate_comprehensive_graph_report(topology_graph: NetworkTopologyGraph,
                                      sequences: Dict[str, Any],
                                      parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Génère un rapport complet sur le graphe topologique entier
    """
    
    stats = topology_graph.get_graph_stats()
    graph = topology_graph.get_graph()
    
    # Analyse avancée du graphe
    node_degrees = dict(graph.degree())
    edge_labels = [data.get('label', 'unknown') for _, _, data in graph.edges(data=True)]
    node_labels = [data.get('label', 'unknown') for _, data in graph.nodes(data=True)]
    
    # Statistiques des labels
    from collections import Counter
    node_label_counts = Counter(node_labels)
    edge_label_counts = Counter(edge_labels)
    
    report = {
        "graph_statistics": {
            **stats,
            "avg_node_degree": sum(node_degrees.values()) / len(node_degrees) if node_degrees else 0,
            "max_node_degree": max(node_degrees.values()) if node_degrees else 0,
            "density": stats["num_edges"] / max(stats["num_nodes"] * (stats["num_nodes"] - 1), 1)
        },
        "sequence_analysis": {
            "general_sequence_length": sequences.get("general", {}).get("length", 0),
            "complete_sequence_length": sequences.get("complete", {}).get("length", 0),
            "user_sequences_count": len(sequences.get("by_user", {})),
            "temporal_span": sequences.get("temporal_stats", {}),
            "unique_users": list(sequences.get("by_user", {}).keys())[:10]  # Top 10
        },
        "content_analysis": {
            "top_node_types": dict(node_label_counts.most_common(10)),
            "top_edge_types": dict(edge_label_counts.most_common(10)),
            "unique_node_types": len(node_label_counts),
            "unique_edge_types": len(edge_label_counts)
        },
        "processing_summary": {
            "total_packets_processed": stats["central_nodes"],
            "data_elements_extracted": stats["leaf_nodes"],
            "relationships_created": stats["num_edges"],
            "graph_complexity": stats["num_nodes"] + stats["num_edges"]
        }
    }
    
    logger.info(f" Rapport complet généré:")
    logger.info(f"   {stats['central_nodes']} paquets → {stats['num_nodes']} nœuds")
    logger.info(f"   {stats['num_edges']} relations créées")
    logger.info(f"   {len(sequences.get('by_user', {}))} utilisateurs analysés")
    logger.info(f"   {len(node_label_counts)} types de nœuds différents")
    
    return report

