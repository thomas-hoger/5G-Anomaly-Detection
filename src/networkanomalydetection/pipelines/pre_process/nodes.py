import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx

from networkanomalydetection.core.feature.type_split import split_by_type
from networkanomalydetection.core.feature.vectorization import vectorize_nodes
from networkanomalydetection.core.graph.construction import build_graph
from networkanomalydetection.core.graph.sampling import generate_subgraphs
from networkanomalydetection.core.graph.visualization import graph_to_html
from networkanomalydetection.core.trace.cleaning_labelling import process
from networkanomalydetection.core.trace.dissection.dissect_packet import dissect_packets


def trace_cleaning_labelling(pkt_files: dict, evil_ip:str):

    cleaned_pkts_files = {}
    label_df_files = {}

    for file, pkt_loader in pkt_files.items():
        file_csv = file.replace(".pcap",".csv")
        cleaned_pkts_files[file], label_df_files[file_csv] = process(pkt_loader(), evil_ip)

    return cleaned_pkts_files, label_df_files

def trace_dissection(pkt_files: dict, banned_features: list[str], label_dataframe_files: dict):

    dissected_files = {}

    for file, pkt_loader in pkt_files.items():

        csv_file   = file.replace("pcap","csv")
        csv_loader = label_dataframe_files[csv_file]

        json_file = file.replace("pcap","json")

        dissected_files[json_file] = dissect_packets(pkt_loader(), banned_features, csv_loader())

    return dissected_files

def graph_building(dissected_files:dict[str,list[dict]]):

    topology_graph_files = {}

    for file, trace_loader in dissected_files.items():

        pkl_file = file.replace("json","pkl")
        topology_graph_files[pkl_file] = build_graph(trace_loader())

    return topology_graph_files

def graph_visualization(graph_files:dict):

    graph_html_files = {}
    for file, graph_loader in graph_files.items():

        html_file = file.replace("pkl","html")
        graph_html_files[html_file] = graph_to_html(graph_loader())

    return graph_html_files

def feature_splitting(graph_files:dict, identifier_conversion:dict[str:str]):

    words  = []
    floats = []
    for file, graph_loader in graph_files.items():

        new_words, new_floats = split_by_type(graph_loader(), identifier_conversion.keys())

        words += new_words
        floats += new_floats

    return words, floats

def feature_vectorization(graph_files:dict, feature_floats:list[float], feature_words:list[str], identifier_conversion:dict[str:str]):

    edge_embeddings = []
    node_embeddings = []

    vectorized_nodes_files = {}
    all_unique_features    = {}

    for file, graph_loader in graph_files.items():

        vectorized_graph, unique_features = vectorize_nodes(graph_loader(), feature_floats, feature_words, identifier_conversion)
        vectorized_nodes_files[file] = vectorized_graph

        node_embeddings += [np.array(data["embedding"]) for _, data in vectorized_graph.nodes(data=True) if "embedding" in data]
        edge_embeddings += [np.array(data["embedding"]) for _, _, data in vectorized_graph.edges(data=True) if "embedding" in data]

        for key,values in unique_features.items():
            if key not in all_unique_features:
                all_unique_features[key] = []
            all_unique_features[key] = list(set(all_unique_features[key] + values))

    reporting = {
        "number_of_nodes": len(node_embeddings),
        "number_of_edges": len(edge_embeddings),
        "unique_nodes_shape" : list(set([emb.shape[0] for emb in node_embeddings])),
        "unique_edges_shape" : list(set([emb.shape[0] for emb in edge_embeddings])),
        "unique_features" : all_unique_features
    }

    return vectorized_nodes_files, reporting

def graph_sampling(graph_files:dict, window_size:int, window_shift:int):

    all_subgraphs  = []
    subgraph_files = {}
    for file, graph_loader in graph_files.items():

        subgraphs = generate_subgraphs(graph_loader(), window_size, window_shift)
        subgraph_files[file] = subgraphs
        all_subgraphs += subgraphs

    reporting = {
        "number_of_graph": len(all_subgraphs),
        "subgraphs_length" : [len(subgraph) for subgraph in all_subgraphs]
    }

    return subgraph_files, reporting

def graph_vectorization(graph_files:dict, batch_size:int, split_ratio:int):

    data_list = []
    for file, graph_loader in graph_files.items():

        graph_list = graph_loader()
        for graph in graph_list:

            # Remove the attributes specific to central nodes to convert them to tensor
            for n, feature in graph.nodes(data=True):
                feature.pop("is_attack", None)  # None = pas d'erreur si absent
                feature.pop("type", None)

            data = from_networkx(graph, group_node_attrs=["embedding"], group_edge_attrs=["embedding"])
            data_list.append(data)

    split_idx = int(len(data_list) * split_ratio)

    data_loader_1 = DataLoader(data_list[:split_idx], batch_size, shuffle=True)
    data_loader_2 = DataLoader(data_list[split_idx:], batch_size, shuffle=True)

    return data_loader_1, data_loader_2
