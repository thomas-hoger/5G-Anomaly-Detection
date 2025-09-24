from enum import Enum

import networkx as nx
import tqdm

topology_graph = nx.Graph()
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

def find_stream(ip_src: str, ip_dst: str, stream_id: int, stream_response: bool) -> int|None:
    central_node_id = None

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

def packet_to_nodes(dissected_pkt: dict, packet_id: int, merge_stream=False) -> int:
    ip_src = dissected_pkt["common"]["ip_src"]
    ip_dst = dissected_pkt["common"]["ip_dst"]
    stream_id = stream_response = None

    # Try to get the stream in the http2 layer
    if "http2" in dissected_pkt and "stream_id" in dissected_pkt["http2"]:
        stream_id = dissected_pkt["http2"].pop("stream_id")
        stream_response = dissected_pkt["http2"].pop("stream_response")

    # Try to find the central
    central_node_id = find_stream(ip_src, ip_dst, stream_id, stream_response)

    # If the stream is not found, create a new central node
    if central_node_id is None or not merge_stream:
        central_node_id = topology_graph.number_of_nodes()
        topology_graph.add_node(central_node_id, label="", node_type=NodeType.CENTRAL.value, packet_id=packet_id)

    # If the stream is found, we don't want to add the IPs again
    else:
        del dissected_pkt["common"]["ip_src"]
        del dissected_pkt["common"]["ip_dst"]

    # Dont put this values as new nodes, but as attributes of the central node
    topology_graph.nodes[central_node_id]["is_attack"] = dissected_pkt["common"].pop("is_attack")
    topology_graph.nodes[central_node_id]["type"]      = dissected_pkt["common"].pop("type")

    # for common
    for edge_label, param_value in dissected_pkt["common"].items():

        # Check if the node already exists
        parameted_node_id = find_node("label", str(param_value))

        # If the node does not exist, create it
        if not parameted_node_id:
            parameted_node_id = topology_graph.number_of_nodes()
            topology_graph.add_node(parameted_node_id, label=str(param_value), node_type=NodeType.PARAMETER.value, packet_id=packet_id)

        # Connect the parameter node to the central node
        topology_graph.add_edge(central_node_id, parameted_node_id, label=edge_label)

    # for protocols : http2, pfcp ...
    for protocol,layers in dissected_pkt["protocols"].items():

        for i,layer in enumerate(layers):

            for param_name, param_value in layer.items():

                # Check if the node already exists
                parameted_node_id = find_node("label", str(param_value))

                # If the node does not exist, create it
                if not parameted_node_id:
                    parameted_node_id = topology_graph.number_of_nodes()
                    topology_graph.add_node(parameted_node_id, label=str(param_value), node_type=NodeType.PARAMETER.value, packet_id=packet_id)

                edge_label = f"{protocol}{[i]}.{param_name}"

                # Connect the parameter node to the central node
                topology_graph.add_edge(central_node_id, parameted_node_id, label=edge_label)

    return central_node_id

def build_graph(trace:list[dict]) -> nx.Graph:

    global topology_graph  # noqa: PLW0603
    topology_graph = nx.Graph()

    packet_id = 0
    for packet in tqdm.tqdm(trace,total=len(trace), desc="Processing packets for building initial graph"):
        packet_to_nodes(packet, packet_id)
        packet_id += 1

    return topology_graph.copy()
