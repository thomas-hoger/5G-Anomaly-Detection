import re

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from sklearn.mixture import GaussianMixture

from networkanomalydetection.core.graph.construction import NodeType

N_COMPONENTS = 17 # TEMPORAIRE, à déterminer
gmm = None

def init_gmm(float_encountered: list[float]):
    global gmm  # noqa: PLW0603
    gmm = GaussianMixture(n_components=N_COMPONENTS)
    data = np.array(float_encountered).reshape(-1,1)
    gmm.fit(data)

def is_float(value) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False

def vectorize_nodes(graph: nx.Graph, float_encountered:list[float], text_encountered:list[str], identifier_conversion:dict[str:str]):

    # Prepare the GMM
    init_gmm(float_encountered)

    # Prepare the one hot
    unique_words = list(set(text_encountered))
    unique_words += ["others"]
    unique_words += [value for value in identifier_conversion.values()]
    unique_words += list(range(N_COMPONENTS))

    word_mapping = {word: i for i, word in enumerate(unique_words)}
    dimension    = len(unique_words)

    report_dict  = {}

    # For each node, change its embedding
    central_node_ids = [n for n, attr in graph.nodes(data=True) if attr["node_type"] == NodeType.CENTRAL.value]
    for central_node_id in tqdm.tqdm(central_node_ids, total=len(central_node_ids), desc="Vectorizing graph"):

        graph.nodes[central_node_id]["embedding"] = torch.zeros(dimension)
        for _, neighbor, edge_attr in graph.edges(central_node_id, data=True):

            # edges are always text and never id -> one hot
            normalized_edge_label = re.sub(r'\[\d+\]', '', edge_attr["label"]) # Remove indies [0], [1], etc.
            edge_labels = torch.tensor([word_mapping[word] for word in normalized_edge_label.split(".")])
            edge_vects  = F.one_hot(edge_labels, num_classes=dimension)
            edge_attr["embedding"] = edge_vects.to(torch.float32).mean(dim=0) # TODO: add weighted mean

            # Embed a neighbor only if it has not been embedded yet
            neighbor_attr = graph.nodes[neighbor]
            if "embedding" not in neighbor_attr:

                found = False
                for identifier in identifier_conversion.keys():
                    if identifier.lower() in edge_attr["label"].lower() :
                        found = True
                        break

                # if its an id (e.g ip, port...) -> hardcode a vector
                if found:
                    value_to_embed = identifier_conversion[identifier]

                # if its a float -> gmm
                elif is_float(neighbor_attr["label"]):
                    label = float(neighbor_attr["label"])
                    value_to_embed = gmm.predict(np.array([[label]]))[0]

                # if its a text -> one hot
                else:
                    value_to_embed = neighbor_attr["label"]
                    if value_to_embed not in word_mapping:
                        value_to_embed = "others"

                embedding = F.one_hot(torch.tensor(word_mapping[value_to_embed]), num_classes=dimension)
                neighbor_attr["embedding"] = embedding

                # Store the value embeddings
                if normalized_edge_label not in report_dict:
                    report_dict[normalized_edge_label] = []
                if value_to_embed not in report_dict[normalized_edge_label]:
                    report_dict[normalized_edge_label] += [str(value_to_embed)]

    return graph, report_dict
