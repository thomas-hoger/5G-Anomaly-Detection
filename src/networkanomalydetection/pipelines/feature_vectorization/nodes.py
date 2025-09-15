import re

import networkx as nx
import torch
import torch.nn.functional as F
import tqdm

from networkanomalydetection.core.feature_vectorization.vectorizer import (
    embed_float,
    init_gmm,
    is_float,
)
from networkanomalydetection.core.graph_construction.manage import NodeType


def vectorize_graph_node(graph: nx.Graph, float_encountered:list[float], text_encountered:list[str], identifier_list:list[str]) -> tuple[nx.Graph, dict[str, any]]:

    # Prepare the GMM
    init_gmm(float_encountered)

    # Prepare the one hot
    unique_words = list(set(text_encountered))
    unique_words += ["identifiers"]
    word_mapping = {word: i for i, word in enumerate(unique_words)}

    dimension = len(unique_words)

    # For each node, change its embedding
    central_node_ids = [n for n, attr in graph.nodes(data=True) if attr["node_type"] == NodeType.CENTRAL.value]
    for central_node_id in tqdm.tqdm(central_node_ids, total=len(central_node_ids), desc="Vectorizing graph"):

        graph.nodes[central_node_id]["embedding"] = torch.zeros(dimension)
        for _, neighbor, edge_attr in graph.edges(central_node_id, data=True):

            neighbor_attr = graph.nodes[neighbor]

            # Embed a neighbor only if it has not been embedded yet
            if "embedding" not in neighbor_attr:

                found = False
                for identifier in identifier_list:
                    if identifier.lower() in edge_attr["label"].lower() :
                        found = True

                # if its an id (e.g ip, port...) -> hardcode a vector
                if found:
                    neighbor_attr["embedding"] = F.one_hot(torch.tensor(word_mapping["identifiers"]), num_classes=dimension)

                # if its a float -> gmm
                elif is_float(neighbor_attr["label"]):
                    neighbor_attr["embedding"] = embed_float(float(neighbor_attr["label"]), dimension)

                # if its a text -> one hot
                else:
                    neighbor_attr["embedding"] = F.one_hot(torch.tensor(word_mapping[neighbor_attr["label"]]), num_classes=dimension)

            # edges are always text and never id -> one hot
            normalized = re.sub(r'\[\d+\]', '', edge_attr["label"]) # Remove indies [0], [1], etc.
            edge_labels = torch.tensor([word_mapping[word] for word in normalized.split(".")])
            edge_vects  = F.one_hot(edge_labels, num_classes=dimension)

            edge_attr["embedding"] = edge_vects.mean(dim=0) # TODO: add weighted mean

    return graph
