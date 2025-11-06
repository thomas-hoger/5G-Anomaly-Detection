import networkx as nx
import torch
import torch.nn.functional as F

from networkanomalydetection.core.graph.construction import NodeType
from networkanomalydetection.core.vocabulary_making import normalize_edge


def vectorize_features(graph: nx.Graph, text_vocabulary:list[str]):

    # Get the list of unique labels
    unique_labels    = list(set(text_vocabulary))
    unique_labels   += ["others"]

    word_mapping  = {word: i for i, word in enumerate(unique_labels)}
    dimension     = len(unique_labels)
    labels_report = {}

    line_graph = nx.line_graph(graph)

    for edge, attrs in graph.edges.items():
        label = attrs["label"]
        line_graph.nodes[edge]["label"] = label

        normalized_edge = normalize_edge(label) # Remove indies [0], [1], etc.
        edge_labels = torch.tensor([word_mapping[word] for word in normalized_edge.split(".")]) # Embed each word separately
        edge_vects  = F.one_hot(edge_labels, num_classes=dimension) # Get 1 tensor for each word
        line_graph.nodes[edge]["embedding"] = edge_vects.to(torch.float32).mean(dim=0) # TODO: add weighted mean

        u,v = edge
        central_node = u if graph.nodes[u].get("node_type") == NodeType.CENTRAL.value else v

        line_graph.nodes[edge]["is_attack"] = graph.nodes[central_node].get("is_attack")
        line_graph.nodes[edge]["type"] = graph.nodes[central_node].get("type")

        # store in the report
        if normalized_edge not in labels_report:
            labels_report[normalized_edge] = set()
        labels_report[normalized_edge].add(label)

    return line_graph, labels_report
