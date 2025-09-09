import networkx as nx
from pyvis.network import Network

from networkanomalydetection.core.graph_construction.manage import NodeType


def graph_to_html(graph: nx.DiGraph) -> str:
    net = Network(
        directed=True,
        height="1000px"
    )

    COLOR_MAP = {
        NodeType.CENTRAL.value: "#B388EB",    # Mauve
        NodeType.PARAMETER.value: "#73C2FB"   # Bleu
    }

    for node_id in graph.nodes:
        graph.nodes[node_id]["color"] = COLOR_MAP.get(graph.nodes[node_id]["node_type"])
        graph.nodes[node_id]["mass"] = graph.degree(node_id)

    net.from_nx(graph)
    return net.generate_html()
