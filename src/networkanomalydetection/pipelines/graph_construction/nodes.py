import networkx as nx
import tqdm

from networkanomalydetection.core.graph_construction.manage import (
    packet_to_nodes,
    topology_graph,
)


def build_graph(trace_dissected:dict[str,list[dict]]) -> str:

    topology_graph_files = {}
    global topology_graph  # noqa: PLW0603

    for file, trace_loader in trace_dissected.items():

        packet_id = 0
        trace = trace_loader()
        for packet in tqdm.tqdm(trace,total=len(trace), desc="Processing packets for building initial graph"):
            packet_to_nodes(packet, packet_id)
            packet_id += 1

        pkl_file = file.replace("pcap","pkl")
        topology_graph_files[pkl_file] = topology_graph.copy()
        topology_graph = nx.Graph()

    return topology_graph_files
