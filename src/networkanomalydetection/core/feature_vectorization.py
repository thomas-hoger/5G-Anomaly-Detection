import re

import networkx as nx
import torch
import torch.nn.functional as F


def vectorize_features(graph: nx.Graph):

    # Get the list of unique labels
    unique_labels = set()
    for _, _, edge_data in graph.edges(data=True):
        if "label" in edge_data:

            # Remove indices [0], [1], etc.
            normalized_edge_label = re.sub(r'\[\d+\]', '', edge_data["label"])
            unique_labels.add(normalized_edge_label)

    unique_labels = list(unique_labels)
    word_mapping  = {word: i for i, word in enumerate(unique_labels)}
    dimension     = len(unique_labels)

    # Change the embedding of the edge with a one hot
    for _, _, edge_data in graph.edges(data=True):
        if "label" in edge_data:

            # Remove indices [0], [1], etc.
            normalized_edge_label = re.sub(r'\[\d+\]', '', edge_data["label"])

            edge_labels = torch.tensor([word_mapping[word] for word in normalized_edge_label.split(".")])
            edge_vects  = F.one_hot(edge_labels, num_classes=dimension)
            edge_data["embedding"] = edge_vects.to(torch.float32).mean(dim=0) # TODO: add weighted mean

    return graph, unique_labels
