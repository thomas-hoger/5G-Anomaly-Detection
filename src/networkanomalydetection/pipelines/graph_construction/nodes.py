import tqdm

from networkanomalydetection.core.graph_construction.manage import (
    packet_to_nodes,
    topology_graph,
)


def build_graph(trace_dissected:list[dict]) -> str:

    packet_id = 0
    for packet in tqdm.tqdm(trace_dissected,total=len(trace_dissected), desc="Processing packets for building initial graph"):
        packet_to_nodes(packet, packet_id)
        packet_id += 1

    return topology_graph
