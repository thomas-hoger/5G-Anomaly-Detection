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

def embed_float(label:float, dimension:int)-> torch.Tensor:
    label_to_array = np.array([[label]])
    prediction     = gmm.predict_proba(label_to_array)[0]
    prediction     = torch.from_numpy(prediction)

    # TODO: TEMPORAIRE, remplacer par fully connected
    padding = torch.zeros(dimension - len(prediction))  # Vecteur de padding
    padded_prediction = torch.cat((prediction, padding))

    return padded_prediction

def vectorize_nodes(graph: nx.Graph, float_encountered:list[float], text_encountered:list[str], identifier_list:list[str]):

    # Prepare the GMM
    init_gmm(float_encountered)

    # Prepare the one hot
    unique_words = list(set(text_encountered))
    unique_words += ["identifiers", "others"]
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
                    embedding = F.one_hot(torch.tensor(word_mapping["identifiers"]), num_classes=dimension)

                # if its a float -> gmm
                elif is_float(neighbor_attr["label"]):
                    embedding = embed_float(float(neighbor_attr["label"]), dimension)
                    embedding = F.pad(embedding, (0, dimension - embedding.size(0)), value=0)

                # if its a text -> one hot
                else:
                    label = neighbor_attr["label"]
                    if label not in word_mapping:
                        label = "others"
                    embedding = F.one_hot(torch.tensor(word_mapping[label]), num_classes=dimension)

                neighbor_attr["embedding"] = embedding

            # edges are always text and never id -> one hot
            normalized = re.sub(r'\[\d+\]', '', edge_attr["label"]) # Remove indies [0], [1], etc.
            edge_labels = torch.tensor([word_mapping[word] for word in normalized.split(".")])
            edge_vects  = F.one_hot(edge_labels, num_classes=dimension)

            edge_attr["embedding"] = edge_vects.to(torch.float32).mean(dim=0) # TODO: add weighted mean

    return graph
