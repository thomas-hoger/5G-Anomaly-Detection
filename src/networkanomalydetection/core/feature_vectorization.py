import networkx as nx
import torch
import torch.nn.functional as F
import tqdm

from networkanomalydetection.core.graph.construction import NodeType
from networkanomalydetection.core.vocabulary_making import (
    find_feature_identifier,
    normalize_edge,
)


def vectorize_features(graph: nx.Graph, identifier_features: dict[str:str], text_vocabulary:list[str]):

    # Get the list of unique labels
    unique_labels    = list(set(text_vocabulary))
    unique_labels   += ["others"]

    word_mapping  = {word: i for i, word in enumerate(unique_labels)}
    dimension     = len(unique_labels)

    labels_report = {}

    central_node_ids = [n for n, attr in graph.nodes(data=True) if attr["node_type"] == NodeType.CENTRAL.value]
    for central_node_id in tqdm.tqdm(central_node_ids, total=len(central_node_ids), desc="Vectorizing graph"):

        # The central node have a ZERO vector to only get the value of its neighbors
        graph.nodes[central_node_id]["embedding"] = torch.zeros(dimension)

        # For every neighbor we change the edge and PARAMETER
        for _, neighbor, edge_attr in graph.edges(central_node_id, data=True):

            # First, the edge embedding
            normalized_edge = normalize_edge(edge_attr["label"]) # Remove indies [0], [1], etc.
            edge_labels = torch.tensor([word_mapping[word] for word in normalized_edge.split(".")]) # Embed each word separately
            edge_vects  = F.one_hot(edge_labels, num_classes=dimension) # Get 1 tensor for each word
            edge_attr["embedding"] = edge_vects.to(torch.float32).mean(dim=0) # TODO: add weighted mean

            # Next, for the parameter, embed it only if it has not been embedded yet
            if "embedding" not in graph.nodes[neighbor]:

                found_feature = find_feature_identifier(graph.nodes[neighbor]["label"], identifier_features)
                if found_feature:
                    label = found_feature

                # if the text value has never been seen we put it to "other"
                elif graph.nodes[neighbor]["label"] not in word_mapping:
                    label = "others"

                # just encode the value
                else:
                    label = graph.nodes[neighbor]["label"]

                # Update the neighbor embedding
                embedding = F.one_hot(torch.tensor(word_mapping[label]), num_classes=dimension)
                graph.nodes[neighbor]["embedding"] = embedding

                # store in the report
                if normalized_edge not in labels_report:
                    labels_report[normalized_edge] = set()
                labels_report[normalized_edge].add(label)

    return graph, labels_report
