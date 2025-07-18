"""
This is a boilerplate pipeline 'dissection'
generated using Kedro 0.19.14
"""

import os
import glob
import logging
from typing import Dict, List, Any
from pathlib import Path

from ...utils.dissection_pyshark import dissect_pcap_file

logger = logging.getLogger(__name__)

def dissect_pcap_files(parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Dissèque plusieurs fichiers PCAP en les lisant directement du dossier
    """
    
    dissection_params = parameters.get("dissection", {})
    max_packets = dissection_params.get("max_packets_per_file", 10000)
    ban_list = dissection_params.get("ban_list", [])
    
    # Lire les fichiers PCAP directement du dossier
    pcap_dir = "data/01_raw/pcap_files"
    pcap_files = {}
    
    if not os.path.exists(pcap_dir):
        logger.error(f"Dossier PCAP non trouvé: {pcap_dir}")
        return []
    
    for filepath in glob.glob(os.path.join(pcap_dir, "*.pcap")):
        filename = os.path.basename(filepath)
        pcap_files[filename] = filepath
    
    if not pcap_files:
        logger.error(f"Aucun fichier PCAP trouvé dans {pcap_dir}")
        return []
    
    logger.info(f"Fichiers PCAP trouvés: {list(pcap_files.keys())}")
    
    all_dissected_packets = []
    
    for filename, filepath in pcap_files.items():
        logger.info(f"Dissection du fichier: {filename}")
        
        packets = dissect_pcap_file(
            pcap_file=filepath,
            max_packets=max_packets,
            ban_list=ban_list
        )
        
        for packet in packets:
            packet["file_info"] = {
                "filename": filename,
                "filepath": filepath
            }
        
        all_dissected_packets.extend(packets)
        logger.info(f"Dissection terminée: {len(packets)} paquets extraits de {filename}")
    
    logger.info(f"Dissection globale terminée: {len(all_dissected_packets)} paquets au total")
    return all_dissected_packets

def generate_dissection_report(dissected_packets: List[Dict[str, Any]],
                              parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Génère un rapport d'analyse de la dissection
    """
    
    # Statistiques générales
    total_packets = len(dissected_packets)
    
    # Analyse basique
    unique_sources = set()
    unique_destinations = set()
    request_count = 0
    response_count = 0
    
    for packet in dissected_packets:
        # Statistiques réseau
        common = packet.get("common", {})
        unique_sources.add(common.get("ip_src"))
        unique_destinations.add(common.get("ip_dst"))
        
        # Compter requêtes/réponses
        if common.get("is_request"):
            request_count += 1
        else:
            response_count += 1
    
    # Identifier les sources de trafic élevé
    source_counts = {}
    for packet in dissected_packets:
        src = packet.get("common", {}).get("ip_src")
        if src:
            source_counts[src] = source_counts.get(src, 0) + 1
    
    high_traffic_sources = [src for src, count in source_counts.items() 
                           if count > total_packets * 0.1]
    
    report = {
        "summary": {
            "total_packets": total_packets,
            "unique_sources": len(unique_sources),
            "unique_destinations": len(unique_destinations),
            "request_count": request_count,
            "response_count": response_count
        },
        "network_analysis": {
            "source_distribution": source_counts,
            "high_traffic_sources": high_traffic_sources
        },
        "anomaly_indicators": {
            "high_traffic_sources": high_traffic_sources
        }
    }
    
    logger.info(f"Rapport généré: {total_packets} paquets analysés")
    return report