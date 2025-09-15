import networkx as nx
from pyvis.network import Network

from networkanomalydetection.core.graph_construction.manage import NodeType


def graph_to_html(graph_files: dict[str,nx.Graph]) -> str:

    graph_html_files = {}
    for file, graph_loader in graph_files.items():

        graph  = graph_loader()

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

        net.from_nx(graph)
        html_file = file.replace("pkl","html")

        graph_html_files[html_file] = net.generate_html()

    return graph_html_files
