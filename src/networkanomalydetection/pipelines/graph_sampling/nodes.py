import networkx as nx
import tqdm

from networkanomalydetection.core.graph_construction.manage import NodeType


def generate_subgraphs(graph: nx.Graph, subgraph_history_length: int) -> list[nx.Graph]:

    subgraphs = []
    central_node_ids = [n for n, attr in graph.nodes(data=True) if attr["node_type"] == NodeType.CENTRAL.value]
    for central_node_id in tqdm.tqdm(central_node_ids, total=len(central_node_ids), desc="Extracting subgraphs"):

        # Making a subgraph of radius 2 around the central node
        egograph = nx.generators.ego_graph(graph, central_node_id, radius=2)

        # Keeping only nodes arrived before or at the same time as the central node
        central_packet_id = graph.nodes[central_node_id]["packet_id"]
        selected_nodes = [
            node for node, attr in graph.nodes(data=True)
            if central_packet_id - subgraph_history_length <= attr["packet_id"] <= central_packet_id
        ]
        egograph_temporal = egograph.subgraph(selected_nodes)
        subgraphs.append(egograph_temporal)

    return subgraphs
