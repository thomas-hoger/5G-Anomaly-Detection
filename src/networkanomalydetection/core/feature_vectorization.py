import networkx as nx
import torch
import torch.nn.functional as F

from networkanomalydetection.core.vocabulary_making import normalize_edge


def vectorize_features(graph: nx.Graph, text_vocabulary:list[str]):

    # Get the list of unique labels
    unique_labels    = list(set(text_vocabulary))
    unique_labels   += ["others"]

    word_mapping  = {word: i for i, word in enumerate(unique_labels)}
    dimension     = len(unique_labels)
    labels_report = {}

    # print(f"\n{dimension} dimensions, {len(graph.edges)} edges, {len(graph.nodes)} nodes")

    for node, attrs in graph.nodes.items():

        if "embedding" not in graph.nodes[node]:
            graph.nodes[node]["embedding"] = torch.ones(dimension, dtype=torch.float32)
        if "type" in graph.nodes[node]:
            del graph.nodes[node]["type"]
        if "is_attack" not in graph.nodes[node]:
            graph.nodes[node]["is_attack"] = -1

    for edge, attrs in graph.edges.items():

        label = attrs["label"]

        normalized_edge = normalize_edge(label) # Remove indies [0], [1], etc.
        edge_labels = torch.tensor([word_mapping[word] for word in normalized_edge.split(".")]) # Embed each word separately
        edge_vects  = F.one_hot(edge_labels, num_classes=dimension) # Get 1 tensor for each word
        graph.edges[edge]["embedding"] = edge_vects.to(torch.float32).mean(dim=0) # TODO: add weighted mean

        # store in the report
        if normalized_edge not in labels_report:
            labels_report[normalized_edge] = set()
        labels_report[normalized_edge].add(label)

    return graph, labels_report
