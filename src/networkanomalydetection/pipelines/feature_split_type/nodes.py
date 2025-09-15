import re

import networkx as nx
import tqdm

from networkanomalydetection.core.feature_vectorization.vectorizer import is_float
from networkanomalydetection.core.graph_construction.manage import NodeType


def split_by_type(graph: nx.Graph, identifier_list:list[str]) -> tuple[nx.Graph, dict[str, any]]:

    processed_nodes   = []
    float_encountered = []
    word_encountered  = []

    # Gather all float values to fit the GMM
    central_node_ids = [n for n, attr in graph.nodes(data=True) if attr["node_type"] == NodeType.CENTRAL.value]
    for central_node_id in tqdm.tqdm(central_node_ids, total=len(central_node_ids), desc="Split features by type"):
        for _, neighbor, edge_attr in graph.edges(central_node_id, data=True):

            # Values can either be float, text or identifier (e.g ip, port...)
            if neighbor not in processed_nodes:
                processed_nodes.append(neighbor)

                label = graph.nodes[neighbor]["label"]
                found = False

                # We don't want to consider identifiers (e.g ip, port...) for the GMM or word list
                for identifier in identifier_list:
                    if identifier.lower() in edge_attr["label"].lower() :
                        found = True

                if not found :

                    # Integers
                    if is_float(label):
                        if abs(float(label)) < 100000:  # noqa: PLR2004
                            float_encountered.append(float(label))

                    # Text
                    else :
                        word_encountered.append(label)

            # Edges are always text and never id (and they can be words separated by dots)
            normalized = re.sub(r'\[\d+\]', '', edge_attr["label"] )
            words = normalized.split(".")
            word_encountered += words

    return word_encountered, float_encountered
