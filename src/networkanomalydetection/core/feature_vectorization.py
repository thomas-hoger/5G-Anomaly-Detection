import networkx as nx
import torch
import torch.nn.functional as F
import tqdm
from torch_geometric.data import Data

from networkanomalydetection.core.graph.construction import NodeType
from networkanomalydetection.core.vocabulary_making import normalize_edge

packet_types = [
    'register_random_ue',
    'add_random_nf',
    'restart',
    'deregister_random_ue',
    'cn_mitm',
    'set_random_ue_idle',
    'uplink_wake_random_u',
    'remove_random_nf',
    'user_traffic',
    'unknown',
    'uplink_spoofing',
    'flood_etablishment',
    'pfcp_in_gtp',
    'seid_fuzzing',
    'applicative_scan',
    'fuzz',
    'downlink_wake_random',
    'modify_drop',
    'modify_dupl',
    'flood_deletion'
]

def vectorize_features(graph: nx.Graph, text_vocabulary:list[str]) -> tuple[Data, dict]:

    # Get the list of unique labels
    unique_labels    = list(set(text_vocabulary))
    # unique_labels   += ["others"]

    unique_sub_labels = []
    for label in unique_labels:
        unique_sub_labels += label.split(".")
    unique_sub_labels = list(set(unique_sub_labels))

    word_mapping  = {word: i for i, word in enumerate(unique_sub_labels)}
    dimension     = len(unique_sub_labels)
    labels_report = {}

    data_dict = {}

    ######### Node features #########

    data_dict["x"] = torch.ones((len(graph.nodes), dimension), dtype=torch.float32)
    data_dict["packet_id"] = torch.zeros((len(graph.nodes)), dtype=torch.long)

    data_dict["is_central"] = torch.zeros(len(graph.nodes), dtype=torch.bool)
    data_dict["is_attack"]  = torch.ones(len(graph.nodes), dtype=torch.short)*-1
    data_dict["type"]       = torch.ones(len(graph.nodes), dtype=torch.short)*-1
    # data_dict["label"]     = []

    # For each nodes
    for i, (node, attrs) in tqdm.tqdm(enumerate(graph.nodes.items()), desc="Node vectorization", total=graph.number_of_nodes()):

        # Is attack
        if "is_attack" in graph.nodes[node]:
            data_dict["is_attack"][i] = int(graph.nodes[node]["is_attack"])

        if "type" in graph.nodes[node]:
            data_dict["type"][i]       = packet_types.index(graph.nodes[node]["type"])

        data_dict["packet_id"][i]  = graph.nodes[node]["packet_id"]
        data_dict["is_central"][i] = graph.nodes[node]["node_type"] == NodeType.CENTRAL.value
        # data_dict["label"].append(graph.nodes[node]["label"])

    ######### Edge features #########

    mapping = dict(zip(graph.nodes(), range(graph.number_of_nodes())))
    data_dict["edge_index"] = torch.empty((2, graph.number_of_edges()), dtype=torch.long)
    data_dict["edge_attr"]  = torch.empty((graph.number_of_edges(), dimension), dtype=torch.float32)
    # data_dict["edge_label"] = []

    # For each edges
    for i, (edge, attrs) in tqdm.tqdm(enumerate(graph.edges.items()), desc="Edge vectorization", total=graph.number_of_edges()):

        src,dst = edge
        data_dict["edge_index"][0, i] = mapping[src]
        data_dict["edge_index"][1, i] = mapping[dst]

        # data_dict["edge_label"].append(attrs["label"])

        # Vectorize the edge label
        normalized_edge = normalize_edge(attrs["label"]) # Remove indies [0], [1], etc.
        edge_labels = torch.tensor([word_mapping[word] for word in normalized_edge.split(".")]) # Embed each word separately
        edge_vects  = F.one_hot(edge_labels, num_classes=dimension) # Get 1 tensor for each word
        data_dict["edge_attr"][i] = edge_vects.to(torch.float32).mean(dim=0) # TODO: add weighted mean

        # store in the report
        if normalized_edge not in labels_report:
            labels_report[normalized_edge] = set()
        labels_report[normalized_edge].add(attrs["label"])

    graph_data = Data.from_dict(data_dict)

    return graph_data, labels_report
