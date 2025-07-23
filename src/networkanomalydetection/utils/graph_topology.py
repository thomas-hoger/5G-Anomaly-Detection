"""
Module de construction de graphe topologique pour les paquets réseau (version simplifiée)
"""

import networkx as nx
from typing import Dict, List, Tuple, Any, Optional

class NetworkTopologyGraph:
    """
    Classe pour gérer le graphe topologique des paquets réseau
    """
    
    def __init__(self):
        self.topology_graph = nx.MultiDiGraph()
        self.pending_requests = {}
    
    def add_packet(self, central_node_label: str, dissected_pkt: dict) -> int:
        """
        Adds a packet to the topology graph by creating a central node and recursively parsing the packet data.
        Args:
            central_node_label (str): The label for the central node.
            dissected_pkt (dict): The dissected packet data in dictionary form.
        Returns:
            int: The ID of the central node created in the topology graph.
        """
        # Those attributes are made to control the graph and are not inherently packet attributes 
        is_request = dissected_pkt["common"].pop("is_request")
        packet_id  = dissected_pkt["common"].pop("packet_id")
        original_request_id = dissected_pkt["common"].pop("original_request_id")

        central_node_id = None

        # if the packet is the continuation of another packet we don't create another central node 
        if original_request_id and is_request: 
            central_node_id = self.find_node(original_request_id,"packet_id")

            # If original_request_id then current packet is the continuation of a precedent packet
            # So the precedent must exist and if it couldn't be found it's an error
            if central_node_id == None : 
                print(f"Error : searching node with packet_id {original_request_id}. Supposed to exist")

        # If not found or not sequel
        if central_node_id == None : 
            central_node_id = self.topology_graph.number_of_nodes()
            self.topology_graph.add_node(central_node_id, label=central_node_label, ts=dissected_pkt["common"]["ts"], is_request=is_request, packet_id=packet_id)

        # Add UE_ID as an attribute to the node to make sequences later
        if "http2" in dissected_pkt and "ue_id" in dissected_pkt["http2"]: 
            ue_id = dissected_pkt["http2"]["ue_id"]
            self.topology_graph.nodes[central_node_id]["ue_id"] = ue_id

        # Recursively parse the json
        self.json_to_node(central_node_id, dissected_pkt.copy())

        return central_node_id

    def find_node(self, node_value: str, attribute: str) -> Optional[int]:
        # Find the node if it already exist
        for node_id, node_data in self.topology_graph.nodes(data=True):
            if attribute in node_data and node_data[attribute] == node_value:
                return node_id
        return None

    def find_edge(self, parent_node: str, child_node_id: str, attribute: str, value: str) -> bool:
        # Find the node if it already exist
        edges = self.topology_graph.get_edge_data(parent_node, child_node_id)
        if edges :
            for edge_id,edge in edges.items():
                if attribute in edge and edge[attribute] == value:
                    return True
        return False

    def is_float(self, text: str) -> bool:
        """
        Vérifie si le texte est un nombre flottant
        """
        try:
            float(text)
            return True
        except ValueError:
            return False

    def is_id(self, field_name: str) -> bool:
        """
        Vérifie si le champ est un identifiant
        """
        id_fields = ['id', 'packet_id', 'stream_id', 'ue_id', 'user_id']
        return any(id_field in field_name.lower() for id_field in id_fields)

    def predict_cluster(self, value: float) -> str:
        """
        Fonction simple de clustering basée sur des ranges
        """
        if value < 100:
            return "small_values"
        elif value < 1000:
            return "medium_values"
        elif value < 10000:
            return "large_values"
        else:
            return "very_large_values"

    def json_to_node(self, parent_node, json_content: Dict[str, Any], var_name: str = ""): 
        """
        Recursively parse a json file and convert it to Node objects each knowing their neighbours. 
        """
        if json_content : 

            if isinstance(json_content, list):
                for new_json_content in json_content : 
                    if new_json_content : 
                        self.json_to_node(parent_node, new_json_content, var_name)

            elif isinstance(json_content, dict):
                for key,new_json_content in json_content.items() :

                    if key in ["http2","common"] : new_var_name = "" # arbitrary skip
                    else : new_var_name = f"{var_name}.{key}" if var_name else key
                    self.json_to_node(parent_node, new_json_content, var_name=new_var_name)
            else:
                node_src_label = self.topology_graph.nodes[parent_node]["label"]
                edge_label     = var_name
                node_dst_label = str(json_content)

                # Si la valeur est un float, on la remplace par son cluster
                if self.is_float(node_dst_label) and not self.is_id(var_name):
                    node_dst_label = self.predict_cluster(float(node_dst_label))

                # Si le nœud et l'arête existent déjà, on les ignore
                child_node_id = self.find_node(node_dst_label, "label") # get id if does with same label exist, else None
                found_edge    = self.find_edge(parent_node, child_node_id, "label", edge_label)

                if not (child_node_id and found_edge) : 
                    
                    # Créer le nœud enfant s'il n'existe pas
                    if not child_node_id: 
                        child_node_id = self.topology_graph.number_of_nodes()
                        self.topology_graph.add_node(child_node_id, label=node_dst_label)

                    # Créer l'arête
                    self.topology_graph.add_edge(parent_node, child_node_id, label=edge_label)

    def get_sequence(self, seq_len: int, general: bool = False, specific_ue_id=None) -> List[Tuple[int, int]]:
        """
        Retrieve a sequence of nodes from the topology graph based on the specified criteria.
        Parameters:
            seq_len (int): The length of the sequence to return.
            general (bool, optional): If True, only nodes without a UE_id will be included. Defaults to False.
            specific_ue_id (optional): If provided, only nodes with the specified UE_id will be included. Defaults to None.
        Returns:
            list: A list of tuples containing node IDs and their timestamps, sorted by timestamp. The length of the list will be equal to seq_len.
        Raises:
            ValueError: If both general and specific_ue_id are provided.
        Notes:
        - If both general and specific_ue_id are not provided, all nodes will be included.
        """
        if general and specific_ue_id : 
            raise ValueError(f"general={general} and specific_ue_id={specific_ue_id}. Only one can be true")

        filtered_nodes = []
        for node, data in self.topology_graph.nodes(data=True):

            if "ts" in data : # central node

                # Messages with the same UE_id 
                if specific_ue_id :
                    if "ue_id" in data and data["ue_id"] == specific_ue_id: 
                        filtered_nodes.append((node, data["ts"]))

                # Only messages without UE_id  
                elif general :
                    if "ue_id" not in data : 
                        filtered_nodes.append((node, data["ts"]))

                # Everything
                else : 
                    filtered_nodes.append((node, data["ts"]))

        sorted_nodes = sorted(filtered_nodes, key=lambda x: x[1])
        return sorted_nodes[len(sorted_nodes)-seq_len:] # Return list of tuple [(id,ts)]

    def get_graph(self) -> nx.MultiDiGraph:
        """Retourne le graphe topologique"""
        return self.topology_graph
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du graphe"""
        return {
            "num_nodes": self.topology_graph.number_of_nodes(),
            "num_edges": self.topology_graph.number_of_edges(),
            "central_nodes": len([n for n, d in self.topology_graph.nodes(data=True) if "ts" in d]),
            "leaf_nodes": len([n for n, d in self.topology_graph.nodes(data=True) if "ts" not in d])
        }


# Fonctions utilitaires pour compatibilité avec l'ancien code si nécessaire
def create_topology_graph() -> NetworkTopologyGraph:
    """Factory function pour créer un graphe topologique"""
    return NetworkTopologyGraph()