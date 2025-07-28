"""
This is a boilerplate pipeline 'graph_construction'
generated using Kedro 1.0.0rc1
"""
"""
Nœuds pour la construction du graphe topologique à partir du JSON complet
"""

import json
from networkanomalydetection.core.graph_construction.graph_topology import packet_to_nodes, topology_graph, display_graph

def process_trace_file(input_data) -> str:
    """
    Traite le fichier de trace
    """
    # Gérer les différents types d'entrée
    if isinstance(input_data, str):
        data = json.loads(input_data)
    else:
        data = input_data
    
    packet_id = 0
    for packet in data:
        for layer in packet:
            if layer:
                central_node_id = packet_to_nodes(layer, packet_id)
                packet_id += 1
                print(f"Central Node ID: {central_node_id}")
            
        if packet_id > 50:  # Limit to first 50 packets for testing
            break
    
    # Afficher le graphe
    display_graph(topology_graph, "data/08_reporting/graph_display2")
    
    return "Graphe genere avec succes dans graph_display2.html"