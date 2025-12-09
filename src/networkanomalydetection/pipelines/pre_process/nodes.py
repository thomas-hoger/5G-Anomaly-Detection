import networkx as nx
import torch
import tqdm
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from networkanomalydetection.core.dissection.dissect_packet import dissect_packets
from networkanomalydetection.core.dissection_clean import dissection_clean
from networkanomalydetection.core.dissection_clusterize import clusterize
from networkanomalydetection.core.feature_vectorization import vectorize_features
from networkanomalydetection.core.graph.construction import build_graph
from networkanomalydetection.core.graph.sampling import generate_subgraphs
from networkanomalydetection.core.graph.visualization import graph_to_html
from networkanomalydetection.core.trace_cleaning_labelling import process
from networkanomalydetection.core.vocabulary_making import get_vocabulary


def trace_cleaning_labelling(pkt_files: dict, evil_ip:str):

    cleaned_pkts_files = {}
    label_df_files = {}

    for file, pkt_loader in pkt_files.items():
        file_csv = file.replace(".pcap",".csv")
        cleaned_pkts_files[file], label_df_files[file_csv] = process(pkt_loader(), evil_ip)

    return cleaned_pkts_files, label_df_files

def trace_dissection(pkt_files: dict, label_dataframe_files: dict):

    dissected_files = {}

    for file, pkt_loader in pkt_files.items():

        csv_file   = file.replace("pcap","csv")
        csv_loader = label_dataframe_files[csv_file]

        json_file = file.replace("pcap","json")

        dissected_files[json_file] = dissect_packets(pkt_loader(), csv_loader())

    return dissected_files

def dissection_cleaning(dissected_files:dict[str,list[dict]], banned_features: list[str]):

    dissected_clean_files = {}
    for file, trace_loader in dissected_files.items():

        dissected_clean_files[file] = dissection_clean(trace_loader(), banned_features)

    return dissected_clean_files

def vocabulary_making(dissected_files:dict[str,list[dict]], identifier_features:dict[str:str], nb_cluster:int):

    words  = {}
    floats = {}

    for file, trace_loader in dissected_files.items():

        new_words, new_floats = get_vocabulary(trace_loader(), identifier_features, nb_cluster)
        words[file]  = list(set(new_words))
        floats[file] = list(set(new_floats))

    return words, floats

def dissection_clusterize(dissected_files:dict[str,list[dict]], float_files:dict, nb_cluster:int):

    float_list = []
    for _, loader in float_files.items():
        float_list += loader()

    cluster_files = {}

    for file, trace_loader in dissected_files.items():

        cluster_files[file] = clusterize(trace_loader(),float_list, nb_cluster)

    return cluster_files

def graph_building(dissected_files:dict[str,list[dict]]):

    traces = []
    for file, trace_loader in dissected_files.items():
        traces += trace_loader()

    graph = build_graph(traces)
    return {"graph.pkl" : graph}

def graph_visualization(graph_files:dict):

    graph_html_files = {}
    for file, graph_loader in graph_files.items():

        html_file = file.replace("pkl","html")
        graph_html_files[html_file] = graph_to_html(graph_loader())

    return graph_html_files

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

def feature_vectorization(graph_files:dict, word_files:dict, split_ratio:int):

    feature_words = []
    for _, loader in word_files.items():
        feature_words += loader()

    reporting_files = {}

    for file, graph_loader in graph_files.items():

        graph = graph_loader()
        max_packet_id = max(nx.get_node_attributes(graph, "packet_id").values())
        packet_id_threshold = int(max_packet_id * split_ratio)

        # Train graph vectorization
        train_nodes = [
            node for node, attr in graph.nodes(data=True)
            if attr["packet_id"] <= packet_id_threshold
        ]
        graph_train = graph.subgraph(train_nodes)
        data_train, unique_features_train = vectorize_features(graph_train, feature_words)

        # Val graph vectorization
        validation_nodes = [
            node for node, attr in graph.nodes(data=True)
            if attr["packet_id"] > packet_id_threshold
        ]
        graph_validation = graph.subgraph(validation_nodes)
        data_val, unique_features_val = vectorize_features(graph_validation, feature_words)

        reporting = {
            "number_of_nodes_train" : len(graph_train.nodes),
            "number_of_edges_train" : len(graph_train.edges),
            "unique_features_train" : len(unique_features_train),
            "number_of_nodes_val"   : len(graph_validation.nodes),
            "number_of_edges_val"   : len(graph_validation.edges),
            "unique_features_val"   : len(unique_features_val)
        }

        reporting_files[file] = reporting

    return data_train, data_val, reporting_files

def graph_vectorization(graph_files:dict, batch_size:int, split_ratio:int):

    # for loop but in reality only 1 file in input (so just return after the first iteration)
    for file, graph_loader in tqdm.tqdm(graph_files.items(), desc="Graph vectorization", unit="graph", total=len(graph_files)):

        graph = graph_loader()
        max_packet_id = max(nx.get_node_attributes(graph, "packet_id").values())
        packet_id_threshold = int(max_packet_id * split_ratio)

        train_nodes = [
            node for node, attr in graph.nodes(data=True)
            if attr["packet_id"] <= packet_id_threshold
        ]

        graph_train = graph.subgraph(train_nodes)
        data_train = from_networkx(graph_train, group_node_attrs=None, group_edge_attrs=["embedding"])
        embedding_size = data_train.edge_attr.size[1]
        data_train.x = torch.ones((len(graph_train.nodes), embedding_size), dtype=torch.float32)

        validation_nodes = [
            node for node, attr in graph.nodes(data=True)
            if attr["packet_id"] > packet_id_threshold
        ]


        graph_validation = graph.subgraph(validation_nodes)
        data_validation = from_networkx(graph_validation, None, group_edge_attrs=["embedding"])
        data_validation.x = torch.ones((len(graph_validation.nodes), embedding_size), dtype=torch.float32)

        return data_train, data_validation
