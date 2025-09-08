"""
Convertisseur principal NetworkX vers PyTorch Geometric
"""
import networkx as nx
import logging
from typing import Dict, Any, Tuple

from .data_extractor import GraphDataExtractor
from .pytorch_builder import PyTorchGeometricBuilder
from .validator import ConversionValidator

logger = logging.getLogger(__name__)

class NetworkXToTorchConverter:
    """Convertisseur principal NetworkX → PyTorch Geometric"""
    
    def __init__(self, validate: bool = True, device: str = 'cpu'):
        """
        Initialisation du convertisseur
        
        Args:
            validate: Effectuer la validation de conversion
            device: Device PyTorch ('cpu' ou 'cuda')
        """
        self.validate_conversion = validate
        self.device = device
        
        # Composants de conversion
        self.data_extractor = GraphDataExtractor()
        self.pytorch_builder = PyTorchGeometricBuilder(device=device)
        self.validator = ConversionValidator() if validate else None
    
    def convert(self, networkx_graph: nx.MultiDiGraph) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Conversion complète NetworkX → PyTorch Geometric
        
        Args:
            networkx_graph: Graphe NetworkX vectorisé
            
        Returns:
            Tuple (pytorch_data, metadata, validation_report)
        """
        logger.info(f"Démarrage conversion graphe: {networkx_graph.number_of_nodes()} nœuds, {networkx_graph.number_of_edges()} arêtes")
        
        try:
            # Étape 1: Extraction des données
            logger.info("Étape 1: Extraction des données NetworkX")
            extracted_data = self.data_extractor.extract(networkx_graph)
            
            # Étape 2: Construction PyTorch Geometric
            logger.info("Étape 2: Construction PyTorch Geometric")
            pytorch_result = self.pytorch_builder.build(extracted_data)
            
            # Étape 3: Validation (optionnelle)
            validation_report = {}
            if self.validate_conversion and self.validator:
                logger.info("Étape 3: Validation de la conversion")
                validation_report = self.validator.validate(networkx_graph, pytorch_result)
            else:
                logger.info("Étape 3: Validation désactivée")
                validation_report = {'overall_status': 'SKIPPED', 'message': 'Validation désactivée'}
            
            # Métadonnées complètes
            complete_metadata = {
                **pytorch_result['metadata'],
                'conversion_config': {
                    'validation_enabled': self.validate_conversion,
                    'device': self.device
                },
                'mapping_info': {
                    'node_mapping_size': len(pytorch_result['node_mapping']),
                    'reverse_mapping_size': len(pytorch_result['reverse_mapping'])
                }
            }
            
            logger.info("Conversion terminée avec succès")
            return pytorch_result['data'], complete_metadata, validation_report
            
        except Exception as e:
            logger.error(f"Erreur lors de la conversion: {str(e)}")
            raise RuntimeError(f"Échec de la conversion NetworkX → PyTorch Geometric: {str(e)}")
    
    def get_conversion_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de conversion"""
        return {
            'extractor_stats': getattr(self.data_extractor, 'stats', {}),
            'builder_stats': getattr(self.pytorch_builder, 'stats', {}),
            'validator_stats': getattr(self.validator, 'stats', {}) if self.validator else {}
        }
