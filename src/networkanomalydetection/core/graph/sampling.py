import networkx as nx
import tqdm


def generate_subgraphs(graph: nx.Graph, window_size:int, window_shift:int) -> list[nx.Graph]:

    max_packet_id = max(nx.get_node_attributes(graph, "packet_id").values())
    subgraph_count = max(((max_packet_id - window_size) // window_shift),0) + 1

    subgraphs = []

    for i in tqdm.tqdm(range(subgraph_count), total=subgraph_count, desc="Graph sampling"):

        nodes_to_remove = [node for node, degree in graph.degree() if degree > 0.5 * (len(graph) - 1)]

        # Get the nodes within the window
        selected_nodes = [
            node for node, attr in graph.nodes(data=True)
            if i*window_shift <= attr["packet_id"] <= i*window_shift + window_size
            and node not in nodes_to_remove
        ]
        subgraph = graph.subgraph(selected_nodes)
        subgraphs.append(subgraph)

    return subgraphs
