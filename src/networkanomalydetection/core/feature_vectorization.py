import networkx as nx
import torch
import torch.nn.functional as F
import tqdm
from torch_geometric.data import Data

from networkanomalydetection.core.vocabulary_making import normalize_edge


def vectorize_features(graph: nx.Graph, text_vocabulary:list[str]) -> tuple[Data, dict]:

    # Get the list of unique labels
    unique_labels    = list(set(text_vocabulary))
    unique_labels   += ["others"]

    word_mapping  = {word: i for i, word in enumerate(unique_labels)}
    dimension     = len(unique_labels)
    labels_report = {}

    data_dict = {}

    ######### Node features #########

    data_dict["x"] = torch.ones((len(graph.nodes), dimension), dtype=torch.float32)
    data_dict["packet_id"] = torch.zeros((len(graph.nodes)), dtype=torch.long)

    data_dict["is_attack"] = []
    data_dict["type"]      = []
    data_dict["label"]     = []

    # For each nodes
    for i, (node, attrs) in tqdm.tqdm(enumerate(graph.nodes.items()), desc="Node vectorization", total=graph.number_of_nodes()):

        # Is attack
        if "is_attack" in graph.nodes[node]:
            data_dict["is_attack"].append(graph.nodes[node]["is_attack"])
        else :
            data_dict["is_attack"].append(-1)

        # Type
        if "type" in graph.nodes[node]:
            data_dict["type"].append(graph.nodes[node]["type"])
        else :
            data_dict["type"].append("unknown")

        data_dict["packet_id"][i] = graph.nodes[node]["packet_id"]
        data_dict["label"].append(graph.nodes[node]["label"])

    ######### Edge features #########

    mapping = dict(zip(graph.nodes(), range(graph.number_of_nodes())))
    data_dict["edge_index"] = torch.empty((2, graph.number_of_edges()), dtype=torch.long)
    data_dict["edge_attr"]  = torch.empty((graph.number_of_edges(), dimension), dtype=torch.float32)
    data_dict["edge_label"] = []

    # For each edges
    for i, (edge, attrs) in tqdm.tqdm(enumerate(graph.edges.items()), desc="Edge vectorization", total=graph.number_of_edges()):

        src,dst = edge
        data_dict["edge_index"][0, i] = mapping[src]
        data_dict["edge_index"][1, i] = mapping[dst]

        data_dict["edge_label"].append(attrs["label"])

        # Vectorize the edge label
        normalized_edge = normalize_edge(attrs["label"]) # Remove indies [0], [1], etc.
        edge_labels = torch.tensor([word_mapping[word] if word in word_mapping else word_mapping["others"] for word in normalized_edge.split(".")]) # Embed each word separately
        edge_vects  = F.one_hot(edge_labels, num_classes=dimension) # Get 1 tensor for each word
        data_dict["edge_attr"][i] = edge_vects.to(torch.float32).mean(dim=0) # TODO: add weighted mean

        # store in the report
        if normalized_edge not in labels_report:
            labels_report[normalized_edge] = set()
        labels_report[normalized_edge].add(attrs["label"])

    graph_data = Data.from_dict(data_dict)

    return graph_data, labels_report
