"""
This is a boilerplate pipeline 'graph_construction'
generated using Kedro 1.0.0rc1
"""
"""
Nœuds pour la construction du graphe topologique à partir du JSON complet
"""

import json
from networkanomalydetection.core.graph_construction.graph_topology import packet_to_nodes, topology_graph, display_graph,reset_topology_graph,get_clean_graph_copy
import pickle
import networkx as nx

def process_trace_file(input_data) -> str:
    """
    Traite le fichier de trace
    """
    # Réinitialiser le graphe global
    reset_topology_graph()
    # Gérer les différents types d'entrée

    if isinstance(input_data, str):
        data = json.loads(input_data)
    else:
        data = input_data
    # ÉTAPE 1: Traitement pour HTML 
    packet_id = 0
    for packet in data:
        for layer in packet:
            if layer:
                central_node_id = packet_to_nodes(layer, packet_id)
                packet_id += 1
                print(f"Central Node ID: {central_node_id}")
            
        if packet_id > 50:  # Limit to first 50 packets for testing
            break
    
    # Générer la visualisation HTML
    display_graph(topology_graph, "data/reporting/graph_display2")
    html_message = "Graphe genere avec succes dans graph_display2.html"

     # ÉTAPE 2: Continuer le traitement pour le graphe complet
    print("Construction du graphe complet pour pickle...")
    
    # Reprendre le traitement là où on s'était arrêté
    for packet in data[packet_id:]:  # Continuer depuis le paquet 51
        for layer in packet:
            if layer:
                central_node_id = packet_to_nodes(layer, packet_id)
                packet_id += 1
                
                # Afficher le progrès tous les 200 paquets
                if packet_id % 200 == 0:
                    print(f"Traités: {packet_id} paquets")
    
    # Obtenir une copie propre du graphe complet
    clean_complete_graph = get_clean_graph_copy()
    
    print(f"Graphe complet final: {clean_complete_graph.number_of_nodes()} nœuds, {clean_complete_graph.number_of_edges()} arêtes")
    
    # Retourner HTML (50 paquets) + graphe complet propre
    return html_message, clean_complete_graph
