
import networkx as nx
from torch_geometric.data.data import Data
from torch_geometric.utils import from_networkx


def graph_to_tensor(graph: nx.Graph) -> Data:

    data_loader = from_networkx(graph, group_node_attrs=["embedding"], group_edge_attrs=["embedding"])

    return data_loader
