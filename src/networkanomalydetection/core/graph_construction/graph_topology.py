# Ton code manage.py tel quel
import networkx as nx
from enum import Enum
from pyvis.network import Network

topology_graph = nx.MultiDiGraph()
opened_stream = {}

class NodeType(Enum):
    CENTRAL = 1
    PARAMETER = 2

def find_node(attr_name: str, attr_value: str) -> int|None:
    for node_id, attrs in topology_graph.nodes(data=True):
        if attrs.get(attr_name) == attr_value:
            return node_id
    return None

def find_edge(attr_name: str, attr_value: str) -> tuple[int,int]|None:
    for u, v, attrs in topology_graph.edges(data=True):
        if attrs.get(attr_name) == attr_value:
            return (u, v)
    return None

def find_stream(ip_src: str, ip_dst: str, stream_id: int|None, stream_response: bool|None) -> int|None:
    central_node_id = None
    
    if not stream_id or not stream_response:
        return None
    
    # The stream have been opened in the same direction (i.e the message is the continuation of a request)
    if (ip_src, ip_dst) in opened_stream and stream_id in opened_stream[(ip_src, ip_dst)]:
        central_node_id = opened_stream[(ip_src, ip_dst)][stream_id]
    
    # The stream have been opened in the other direction (i.e the message is a response)
    elif (ip_dst, ip_src) in opened_stream and stream_id in opened_stream[(ip_dst, ip_src)]:
        central_node_id = opened_stream[(ip_dst, ip_src)][stream_id]
        
        # If its the end of the stream, remove it from the history
        if stream_response:
            del opened_stream[(ip_dst, ip_src)][stream_id]
    # The packet is the first message of the stream
    else:
        # Initialize the communication history between the two IPs
        if (ip_src, ip_dst) not in opened_stream:
            opened_stream[(ip_src, ip_dst)] = {}
        
        # Add the stream to the history
        opened_stream[(ip_src, ip_dst)][stream_id] = topology_graph.number_of_nodes()
    
    return central_node_id

def packet_to_nodes(dissected_pkt: dict, packet_id: int) -> int:
    ip_src = dissected_pkt["common"]["ip_src"]
    ip_dst = dissected_pkt["common"]["ip_dst"]
    stream_id = stream_response = None
    
    # Try to get the stream in the http2 layer
    if "http2" in dissected_pkt and "stream_id" in dissected_pkt["http2"]:
        stream_id = dissected_pkt["http2"].pop("stream_id")
        stream_response = dissected_pkt["http2"].pop("stream_response")
    
    central_node_id = find_stream(ip_src, ip_dst, stream_id, stream_response)
    
    # If the stream is not found, create a new central node
    if central_node_id is None:
        central_node_id = topology_graph.number_of_nodes()
        topology_graph.add_node(central_node_id, label="", node_type=NodeType.CENTRAL.value, packet_id=packet_id)
    
    # Parameters common, http2, etc...
    every_parameters = {}
    for value in dissected_pkt.values():
        every_parameters.update(value)
    
    # Add the parameter node and edges
    for param_name, param_value in every_parameters.items():
        # Convert the parameter value to string else it wouldnt display on the graph
        param_value = str(param_value)
        
        # Check if the node already exists
        parameted_node_id = find_node("label", param_value)
        
        # If the node does not exist, create it
        if not parameted_node_id:
            parameted_node_id = topology_graph.number_of_nodes()
            topology_graph.add_node(parameted_node_id, label=param_value, node_type=NodeType.PARAMETER.value, packet_id=packet_id)
        
        # Connect the parameter node to the central node
        topology_graph.add_edge(central_node_id, parameted_node_id, label=param_name)
    
    return central_node_id

# Fonction pour afficher le graphe 

def display_graph(graph: nx.DiGraph, output_file: str):
    net = Network(
        directed=True,
        height="1000px"
    )

    # Couleurs personnalisées
    COLOR_MAP = {
        NodeType.CENTRAL.value: "#B388EB",    # Mauve
        NodeType.PARAMETER.value: "#73C2FB"   # Bleu
    }
        
    # IMPORTANT: Créer une copie pour la visualisation
    graph_for_display = graph.copy()
    
    for node_id in graph_for_display.nodes:
        graph_for_display.nodes[node_id]["color"] = COLOR_MAP.get(graph_for_display.nodes[node_id]["node_type"])
        graph_for_display.nodes[node_id]["mass"] = graph_for_display.degree(node_id)

    net.from_nx(graph_for_display)  # Utiliser la copie
    net.save_graph(f'{output_file}.html')

def reset_topology_graph():
    """Réinitialise le graphe global pour un nouveau traitement"""
    global topology_graph, opened_stream
    topology_graph.clear()
    opened_stream.clear()

def get_clean_graph_copy() -> nx.MultiDiGraph:
    """
    Retourne une copie propre du graphe sans les attributs de visualisation
    """
    clean_graph = topology_graph.copy()
    
    # Supprimer les attributs de visualisation si ils existent
    for node_id in clean_graph.nodes():
        attrs_to_remove = ['color', 'mass']  # Attributs de visualisation
        for attr in attrs_to_remove:
            if attr in clean_graph.nodes[node_id]:
                del clean_graph.nodes[node_id][attr]
    
    return clean_graph