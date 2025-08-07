"""
Validateur de conversion NetworkX vers PyTorch Geometric
"""
import numpy as np
import torch
import networkx as nx
import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

class ConversionValidator:
    """Validation de la qualité de conversion"""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate(self, original_graph: nx.MultiDiGraph, 
                converted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validation complète de la conversion
        
        Args:
            original_graph: Graphe NetworkX original
            converted_data: Données converties par PyTorchGeometricBuilder
            
        Returns:
            Rapport de validation complet
        """
        logger.info("Démarrage de la validation de conversion")
        
        torch_data = converted_data['data']
        metadata = converted_data['metadata']
        node_mapping = converted_data['node_mapping']
        
        validation_report = {
            'overall_status': 'PENDING',
            'tests_passed': 0,
            'tests_failed': 0,
            'warnings': [],
            'errors': [],
            'detailed_results': {}
        }
        
        # Tests de validation
        validation_report = self._validate_dimensions(original_graph, torch_data, validation_report)
        validation_report = self._validate_data_integrity(original_graph, torch_data, node_mapping, validation_report)
        validation_report = self._validate_embeddings_quality(original_graph, torch_data, node_mapping, validation_report)
        validation_report = self._validate_graph_structure(original_graph, torch_data, node_mapping, validation_report)
        
        # Statut global
        if validation_report['tests_failed'] == 0:
            validation_report['overall_status'] = 'PASSED'
        elif validation_report['tests_failed'] > validation_report['tests_passed']:
            validation_report['overall_status'] = 'FAILED'
        else:
            validation_report['overall_status'] = 'WARNING'
        
        logger.info(f"Validation terminée: {validation_report['overall_status']} "
                   f"({validation_report['tests_passed']} passed, {validation_report['tests_failed']} failed)")
        
        return validation_report
    
    def _validate_dimensions(self, original_graph: nx.MultiDiGraph, 
                           torch_data: Dict, validation_report: Dict) -> Dict:
        """Validation des dimensions"""
        
        test_name = "dimension_consistency"
        try:
            orig_nodes = original_graph.number_of_nodes()
            orig_edges = original_graph.number_of_edges()
            
            conv_nodes = torch_data['x'].size(0)
            conv_edges = torch_data['edge_index'].size(1)
            
            # Tests
            nodes_match = orig_nodes == conv_nodes
            edges_match = orig_edges == conv_edges
            
            validation_report['detailed_results'][test_name] = {
                'status': 'PASSED' if nodes_match and edges_match else 'FAILED',
                'original_nodes': orig_nodes,
                'converted_nodes': conv_nodes,
                'nodes_match': nodes_match,
                'original_edges': orig_edges,
                'converted_edges': conv_edges,
                'edges_match': edges_match
            }
            
            if nodes_match and edges_match:
                validation_report['tests_passed'] += 1
            else:
                validation_report['tests_failed'] += 1
                error_msg = f"Dimension mismatch: nodes {orig_nodes}->{conv_nodes}, edges {orig_edges}->{conv_edges}"
                validation_report['errors'].append(error_msg)
                
        except Exception as e:
            validation_report['tests_failed'] += 1
            validation_report['errors'].append(f"Erreur validation dimensions: {str(e)}")
            validation_report['detailed_results'][test_name] = {'status': 'ERROR', 'error': str(e)}
        
        return validation_report
    
    def _validate_data_integrity(self, original_graph: nx.MultiDiGraph,
                               torch_data: Dict, node_mapping: Dict, 
                               validation_report: Dict) -> Dict:
        """Validation de l'intégrité des données"""
        
        test_name = "data_integrity"
        try:
            # Vérifier que tous les nœuds ont un mapping
            all_nodes_mapped = len(node_mapping) == original_graph.number_of_nodes()
            
            # Vérifier les indices d'arêtes
            edge_index = torch_data['edge_index']
            max_node_idx = torch.max(edge_index).item()
            valid_indices = max_node_idx < torch_data['num_nodes']
            
            # Vérifier qu'il n'y a pas de valeurs NaN
            no_nan_nodes = not torch.isnan(torch_data['x']).any()
            no_nan_edges = not torch.isnan(torch_data['edge_attr']).any()
            
            all_tests_pass = all([all_nodes_mapped, valid_indices, no_nan_nodes, no_nan_edges])
            
            validation_report['detailed_results'][test_name] = {
                'status': 'PASSED' if all_tests_pass else 'FAILED',
                'all_nodes_mapped': all_nodes_mapped,
                'valid_edge_indices': valid_indices,
                'no_nan_in_node_features': no_nan_nodes,
                'no_nan_in_edge_features': no_nan_edges,
                'max_edge_index': max_node_idx,
                'num_nodes': torch_data['num_nodes']
            }
            
            if all_tests_pass:
                validation_report['tests_passed'] += 1
            else:
                validation_report['tests_failed'] += 1
                validation_report['errors'].append("Problème d'intégrité des données détecté")
                
        except Exception as e:
            validation_report['tests_failed'] += 1
            validation_report['errors'].append(f"Erreur validation intégrité: {str(e)}")
            validation_report['detailed_results'][test_name] = {'status': 'ERROR', 'error': str(e)}
        
        return validation_report
    
    def _validate_embeddings_quality(self, original_graph: nx.MultiDiGraph,
                                   torch_data: Dict, node_mapping: Dict,
                                   validation_report: Dict) -> Dict:
        """Validation de la qualité des embeddings"""
        
        test_name = "embeddings_quality"
        try:
            # Échantillonner quelques nœuds pour vérification
            sample_nodes = list(original_graph.nodes())[:min(100, len(original_graph.nodes()))]
            
            embeddings_match = 0
            total_checked = 0
            
            for orig_node_id in sample_nodes:
                if orig_node_id in node_mapping:
                    torch_idx = node_mapping[orig_node_id]
                    orig_attrs = original_graph.nodes[orig_node_id]
                    
                    if 'embedding' in orig_attrs:
                        orig_embedding = np.array(orig_attrs['embedding'])
                        conv_embedding = torch_data['x'][torch_idx].cpu().numpy()
                        
                        # Comparaison avec tolérance numérique
                        if np.allclose(orig_embedding, conv_embedding, rtol=1e-5, atol=1e-8):
                            embeddings_match += 1
                        
                        total_checked += 1
            
            match_ratio = embeddings_match / max(total_checked, 1)
            quality_ok = match_ratio > 0.95  # 95% des embeddings doivent correspondre
            
            validation_report['detailed_results'][test_name] = {
                'status': 'PASSED' if quality_ok else 'FAILED',
                'embeddings_checked': total_checked,
                'embeddings_matching': embeddings_match,
                'match_ratio': match_ratio,
                'threshold': 0.95
            }
            
            if quality_ok:
                validation_report['tests_passed'] += 1
            else:
                validation_report['tests_failed'] += 1
                validation_report['errors'].append(f"Qualité embeddings insuffisante: {match_ratio:.3f} < 0.95")
                
        except Exception as e:
            validation_report['tests_failed'] += 1
            validation_report['errors'].append(f"Erreur validation embeddings: {str(e)}")
            validation_report['detailed_results'][test_name] = {'status': 'ERROR', 'error': str(e)}
        
        return validation_report
    
    def _validate_graph_structure(self, original_graph: nx.MultiDiGraph,
                                torch_data: Dict, node_mapping: Dict,
                                validation_report: Dict) -> Dict:
        """Validation de la structure du graphe"""
        
        test_name = "graph_structure"
        try:
            # Vérifier quelques connexions aléatoires
            edge_index = torch_data['edge_index'].cpu().numpy()
            sample_edges = min(100, edge_index.shape[1])
            
            structure_matches = 0
            total_checked = 0
            
            # Créer un dictionnaire des arêtes originales pour recherche rapide
            orig_edges = set()
            for u, v in original_graph.edges():
                if u in node_mapping and v in node_mapping:
                    u_idx = node_mapping[u]
                    v_idx = node_mapping[v]
                    orig_edges.add((u_idx, v_idx))
            
            # Vérifier un échantillon d'arêtes converties
            for i in range(sample_edges):
                u_idx = edge_index[0, i]
                v_idx = edge_index[1, i]
                
                if (u_idx, v_idx) in orig_edges:
                    structure_matches += 1
                
                total_checked += 1
            
            structure_ratio = structure_matches / max(total_checked, 1)
            structure_ok = structure_ratio > 0.95
            
            validation_report['detailed_results'][test_name] = {
                'status': 'PASSED' if structure_ok else 'FAILED',
                'edges_checked': total_checked,
                'edges_matching': structure_matches,
                'structure_ratio': structure_ratio,
                'threshold': 0.95
            }
            
            if structure_ok:
                validation_report['tests_passed'] += 1
            else:
                validation_report['tests_failed'] += 1
                validation_report['errors'].append(f"Structure du graphe altérée: {structure_ratio:.3f} < 0.95")
                
        except Exception as e:
            validation_report['tests_failed'] += 1
            validation_report['errors'].append(f"Erreur validation structure: {str(e)}")
            validation_report['detailed_results'][test_name] = {'status': 'ERROR', 'error': str(e)}
        
        return validation_report