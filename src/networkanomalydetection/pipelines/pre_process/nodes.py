import tqdm
from torch_geometric.loader import DataLoader
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

def feature_vectorization(graph_files:dict, word_files:dict):

    feature_words = []
    for _, loader in word_files.items():
        feature_words += loader()

    vectorized_nodes_files = {}
    reporting_files = {}

    for file, graph_loader in graph_files.items():

        graph_list = graph_loader()
        # file_name = file.replace(".pkl", "")

        reporting = {
            "number_of_nodes" : [],
            "number_of_edges" : [],
            "unique_features" : []
        }

        for i,graph in tqdm.tqdm(enumerate(graph_list), desc="Feature vectorization", unit="graph", total=len(graph_list)):

            vectorized_graph, unique_features = vectorize_features(graph, feature_words)
            # subgraph_file_name = f"{file_name}_subgraph_{i}.pkl"
            graph_list[i] = vectorized_graph

            reporting["number_of_nodes"] = len(vectorized_graph.nodes)
            reporting["number_of_edges"] = len(vectorized_graph.edges)
            reporting["unique_features"] = len(unique_features)

        vectorized_nodes_files[file] = graph_list
        reporting_files[file] = reporting

    return vectorized_nodes_files, reporting_files

def graph_vectorization(graph_files:dict, batch_size:int, split_ratio:int):

    data_list = []
    for file, graph_loader in tqdm.tqdm(graph_files.items(), desc="Graph vectorization", unit="graph", total=len(graph_files)):
        for graph in graph_loader():

            data = from_networkx(graph, group_node_attrs=["embedding"], group_edge_attrs=["embedding"])
            data_list.append(data)

    split_idx = int(len(data_list) * split_ratio)

    data_loader_1 = DataLoader(data_list[:split_idx], batch_size, shuffle=False)
    data_loader_2 = DataLoader(data_list[split_idx:], batch_size, shuffle=False)

    return data_loader_1, data_loader_2
