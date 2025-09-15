import networkx as nx
from pyvis.network import Network

from networkanomalydetection.core.graph.construction import NodeType


def graph_to_html(graph: nx.Graph) -> str:

    net = Network(
        directed=False,
        height="1000px"
    )

    COLOR_MAP = {
        NodeType.CENTRAL.value: "#B388EB",    # Mauve
        NodeType.PARAMETER.value: "#73C2FB"   # Bleu
    }

    for node_id in graph.nodes:
        graph.nodes[node_id]["color"] = COLOR_MAP.get(graph.nodes[node_id]["node_type"])
        graph.nodes[node_id]["mass"] = graph.degree(node_id)

    number_of_packets = 10

    central_packet_id = graph.nodes[0]["packet_id"]
    selected_nodes = [
        node for node, attr in graph.nodes(data=True)
        if attr["packet_id"] <= central_packet_id + number_of_packets
    ]
    reduced_graph = graph.subgraph(selected_nodes)

    net.from_nx(reduced_graph)

    return net.generate_html()
