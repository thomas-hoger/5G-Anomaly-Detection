"""
Module pour unifier les dimensions des vecteurs de nœuds et arêtes
"""
import numpy as np
from typing import Union, List
import logging

logger = logging.getLogger(__name__)

class DimensionUnifier:
    """Unifie toutes les dimensions des embeddings à une taille cible"""
    
    def __init__(self, target_dim: int = 64):
        self.target_dim = target_dim
        
    def unify_vector(self, vector: Union[List, np.ndarray], 
                     entity_type: str = "unknown") -> np.ndarray:
        """
        Unifie un vecteur à la dimension cible
        
        Args:
            vector: Vecteur à unifier
            entity_type: Type d'entité pour le logging
            
        Returns:
            np.ndarray: Vecteur unifié à target_dim dimensions
        """
        # Conversion en numpy array
        vector = np.array(vector, dtype=np.float32)
        current_dim = len(vector)
        
        if current_dim == self.target_dim:
            return vector
        elif current_dim > self.target_dim:
            return self._reduce_dimension(vector, entity_type)
        else:
            return self._extend_dimension(vector, entity_type)
    
    def _reduce_dimension(self, vector: np.ndarray, entity_type: str) -> np.ndarray:
        """Réduit la dimension du vecteur"""
        current_dim = len(vector)
        
        # Stratégie : garder les features les plus importantes (valeurs absolues élevées)
        importance_scores = np.abs(vector)
        top_indices = np.argsort(importance_scores)[-self.target_dim:]
        
        reduced_vector = vector[top_indices]
        
        logger.debug(f"Reduced {entity_type} from {current_dim}D to {self.target_dim}D")
        return reduced_vector
    
    def _extend_dimension(self, vector: np.ndarray, entity_type: str) -> np.ndarray:
        """Étend la dimension du vecteur"""
        current_dim = len(vector)
        missing_dims = self.target_dim - current_dim
        
        # Stratégie : padding avec la moyenne des valeurs existantes + petit bruit
        if current_dim > 0:
            mean_val = np.mean(vector)
            std_val = np.std(vector) if current_dim > 1 else 0.1
            
            # Génération déterministe basée sur le contenu
            np.random.seed(hash(str(vector.tolist())) % (2**31))
            padding = np.random.normal(mean_val, std_val * 0.1, missing_dims)
        else:
            # Vecteur vide, padding avec de petites valeurs aléatoires
            np.random.seed(42)
            padding = np.random.normal(0, 0.01, missing_dims)
        
        extended_vector = np.concatenate([vector, padding])
        
        logger.debug(f"Extended {entity_type} from {current_dim}D to {self.target_dim}D")
        return extended_vector
    
    def validate_dimension(self, vector: np.ndarray, entity_type: str) -> bool:
        """Valide que le vecteur a la bonne dimension"""
        is_valid = len(vector) == self.target_dim
        if not is_valid:
            logger.warning(f"Invalid dimension for {entity_type}: {len(vector)} != {self.target_dim}")
        return is_valid
    
    def get_stats(self, vectors: List[np.ndarray]) -> dict:
        """Retourne des statistiques sur les vecteurs"""
        if not vectors:
            return {}
        
        vectors_array = np.array(vectors)
        return {
            'count': len(vectors),
            'mean': np.mean(vectors_array, axis=0),
            'std': np.std(vectors_array, axis=0),
            'min': np.min(vectors_array, axis=0),
            'max': np.max(vectors_array, axis=0),
            'sparsity': np.mean(vectors_array == 0)
        }