"""
Gestionnaire pour les structures hiérarchiques comme nfInstances[0].nfType
"""
import re
import numpy as np
import hashlib
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class HierarchicalPathInfo:
    """Information sur un chemin hiérarchique décomposé"""
    
    def __init__(self, original_path: str):
        self.original_path = original_path
        self.root = ""
        self.hierarchy_levels = []
        self.indices = []
        self.leaf_attribute = ""
        self.parent_path = ""
        self.instance_path = ""
        self.depth = 0
        self.is_hierarchical = False

class HierarchicalPathAnalyzer:
    """Analyseur de chemins hiérarchiques"""
    
    def __init__(self):
        self.path_cache = {}  # Cache pour éviter de recalculer
    
    def analyze_path(self, path: str) -> HierarchicalPathInfo:
        """
        Analyse un chemin hiérarchique
        
        Args:
            path: Chemin à analyser (ex: "nfInstances[0].nfServices[1].serviceName")
            
        Returns:
            HierarchicalPathInfo: Information structurée sur le chemin
        """
        if path in self.path_cache:
            return self.path_cache[path]
        
        info = HierarchicalPathInfo(path)
        
        # Vérifier si c'est hiérarchique
        if '[' not in path or ']' not in path:
            info.is_hierarchical = False
            info.leaf_attribute = path
            self.path_cache[path] = info
            return info
        
        try:
            info.is_hierarchical = True
            
            # Regex pour extraire les composants
            # Pattern: word[index].word[index]...
            pattern = r'([a-zA-Z_][a-zA-Z0-9_]*?)(?:\[(\d+)\])?(?:\.|\Z)'
            matches = re.findall(pattern, path)
            
            hierarchy_levels = []
            indices = []
            
            for match in matches:
                component, index = match
                if component:  # Ignorer les matches vides
                    hierarchy_levels.append(component)
                    if index:
                        indices.append(int(index))
            
            info.hierarchy_levels = hierarchy_levels
            info.indices = indices
            info.root = hierarchy_levels[0] if hierarchy_levels else ""
            info.leaf_attribute = hierarchy_levels[-1] if hierarchy_levels else path
            info.depth = len(hierarchy_levels)
            
            # Construire le chemin parent (sans le dernier attribut)
            if len(hierarchy_levels) > 1:
                info.parent_path = self._construct_parent_path(hierarchy_levels[:-1], indices)
            else:
                info.parent_path = ""
            
            # Construire le chemin d'instance complet
            info.instance_path = self._construct_instance_path(hierarchy_levels, indices)
            
        except Exception as e:
            logger.warning(f"Error analyzing hierarchical path '{path}': {e}")
            info.is_hierarchical = False
            info.leaf_attribute = path
        
        self.path_cache[path] = info
        return info
    
    def _construct_parent_path(self, levels: List[str], indices: List[int]) -> str:
        """Construit le chemin parent"""
        if not levels:
            return ""
        
        parts = []
        index_pos = 0
        
        for level in levels:
            if index_pos < len(indices):
                parts.append(f"{level}[{indices[index_pos]}]")
                index_pos += 1
            else:
                parts.append(level)
        
        return ".".join(parts)
    
    def _construct_instance_path(self, levels: List[str], indices: List[int]) -> str:
        """Construit le chemin d'instance complet"""
        if not levels:
            return ""
        
        parts = []
        index_pos = 0
        
        for level in levels:
            if index_pos < len(indices):
                parts.append(f"{level}[{indices[index_pos]}]")
                index_pos += 1
            else:
                parts.append(level)
        
        return ".".join(parts)

class HierarchicalEdgeEncoder:
    """Encodeur spécialisé pour les arêtes hiérarchiques"""
    
    def __init__(self, target_dim: int = 64):
        self.target_dim = target_dim
        self.path_analyzer = HierarchicalPathAnalyzer()
        self.instance_embeddings_cache = {}  # Cache pour embeddings d'instances
        
        # Embeddings pour concepts racine 5G
        self.root_concept_embeddings = {
            'nfInstances': [1, 0, 0, 0, 0.9],      # Network Function instances
            'nfServices': [0, 1, 0, 0, 0.8],       # Services offered
            'plmnList': [0, 0, 1, 0, 0.7],         # PLMN configurations
            'ipv4Addresses': [0, 0, 0, 1, 0.8],    # Network addresses
            'customInfo': [0, 0, 0, 0, 0.5],       # Custom attributes
            'supportedFeatures': [1, 1, 0, 0, 0.6], # Feature support
        }
    
    def encode_hierarchical_edge(self, edge_key: str) -> np.ndarray:
        """
        Encode une arête hiérarchique
        
        Args:
            edge_key: Clé d'arête hiérarchique
            
        Returns:
            np.ndarray: Vecteur encodé de dimension target_dim
        """
        try:
            # Analyser le chemin
            path_info = self.path_analyzer.analyze_path(edge_key)
            
            if not path_info.is_hierarchical:
                # Pas hiérarchique, encodage simple
                return self._encode_simple_key(edge_key)
            
            # Encodage hiérarchique complet
            components = []
            
            # 1. Encodage du concept racine (16D)
            root_encoding = self._encode_root_concept(path_info.root)
            components.extend(root_encoding)
            
            # 2. Encodage de la structure hiérarchique (16D)
            structure_encoding = self._encode_hierarchy_structure(path_info)
            components.extend(structure_encoding)
            
            # 3. Encodage de l'attribut feuille (16D)
            leaf_encoding = self._encode_leaf_attribute(path_info.leaf_attribute)
            components.extend(leaf_encoding)
            
            # 4. Encodage du contexte d'instance (16D) - CRUCIAL pour cohérence
            instance_encoding = self._encode_instance_context(path_info)
            components.extend(instance_encoding)
            
            # Ajuster à la dimension cible
            return self._adjust_to_target_dim(np.array(components, dtype=np.float32))
            
        except Exception as e:
            logger.warning(f"Error encoding hierarchical edge '{edge_key}': {e}")
            return self._encode_simple_key(edge_key)
    
    def _encode_root_concept(self, root: str) -> List[float]:
        """Encode le concept racine (16D)"""
        base_embedding = self.root_concept_embeddings.get(root, [0, 0, 0, 0, 0.1])
        
        # Étendre à 16 dimensions
        extended = base_embedding.copy()
        
        # Ajouter des features contextuelles
        extended.extend([
            float('nf' in root.lower()),           # Network Function related
            float('service' in root.lower()),      # Service related
            float('list' in root.lower()),         # List/array structure
            float('info' in root.lower()),         # Information field
            float('address' in root.lower()),      # Address field
            len(root) / 20.0,                     # Length normalized
            root.count('s') / max(len(root), 1),  # Plural indicator
            float(root.endswith('s')),            # Ends with 's' (plural)
            float(root.startswith('n')),          # Starts with 'n' (5G convention)
            (hash(root) % 1000) / 1000.0,        # Unique identifier
            0.0  # Padding to reach 16D
        ])
        
        return extended[:16]
    
    def _encode_hierarchy_structure(self, path_info: HierarchicalPathInfo) -> List[float]:
        """Encode la structure hiérarchique (16D)"""
        return [
            path_info.depth / 10.0,                                    # Profondeur normalisée
            len(path_info.indices) / 5.0,                             # Nombre d'indices
            max(path_info.indices) / 100.0 if path_info.indices else 0, # Index maximum
            min(path_info.indices) / 100.0 if path_info.indices else 0, # Index minimum
            np.mean(path_info.indices) / 100.0 if path_info.indices else 0, # Index moyen
            float('Services' in path_info.instance_path),              # Contient services
            float('List' in path_info.instance_path),                  # Contient listes
            float('Info' in path_info.instance_path),                  # Contient info
            float('Address' in path_info.instance_path),               # Contient adresses
            len(path_info.original_path) / 100.0,                     # Longueur totale
            path_info.original_path.count('.') / 10.0,                # Nombre de niveaux
            path_info.original_path.count('[') / 5.0,                 # Nombre d'indices
            float(path_info.depth > 2),                               # Hiérarchie profonde
            float(path_info.depth == 2),                              # Hiérarchie simple
            float(len(path_info.indices) > 1),                        # Multi-indices
            0.0  # Padding
        ]
    
    def _encode_leaf_attribute(self, leaf: str) -> List[float]:
        """Encode l'attribut feuille (16D)"""
        # Embeddings pour attributs courants
        common_attributes = {
            'nfType': [1, 0, 0, 0, 0.9],
            'nfStatus': [0, 1, 0, 0, 0.8],
            'serviceName': [0, 0, 1, 0, 0.9],
            'serviceStatus': [0, 1, 1, 0, 0.8],
            'mcc': [0, 0, 0, 1, 0.7],
            'mnc': [0, 0, 0, 1, 0.7],
            'oauth2': [1, 0, 1, 0, 0.8],
        }
        
        base_embedding = common_attributes.get(leaf, [0, 0, 0, 0, 0.1])
        
        # Étendre avec features spécifiques
        extended = base_embedding.copy()
        extended.extend([
            len(leaf) / 20.0,                     # Longueur
            float('status' in leaf.lower()),      # Statut
            float('name' in leaf.lower()),        # Nom
            float('id' in leaf.lower()),          # Identifiant
            float('type' in leaf.lower()),        # Type
            float('address' in leaf.lower()),     # Adresse
            float('info' in leaf.lower()),        # Information
            float(leaf.endswith('Id')),          # ID field
            float(leaf.startswith('nf')),        # NF field
            (hash(leaf) % 1000) / 1000.0,        # Hash unique
            0.0  # Padding
        ])
        
        return extended[:16]
    
    def _encode_instance_context(self, path_info: HierarchicalPathInfo) -> List[float]:
        """
        Encode le contexte d'instance - CRUCIAL pour cohérence !
        Même instance = même contexte
        """
        instance_key = path_info.parent_path if path_info.parent_path else path_info.instance_path
        
        # Utiliser un cache pour garantir la cohérence
        if instance_key not in self.instance_embeddings_cache:
            self.instance_embeddings_cache[instance_key] = self._generate_stable_instance_embedding(instance_key)
        
        return self.instance_embeddings_cache[instance_key]
    
    def _generate_stable_instance_embedding(self, instance_key: str) -> List[float]:
        """Génère un embedding stable et reproductible pour une instance"""
        if not instance_key:
            return [0.0] * 16
        
        # Hash stable basé sur la clé d'instance
        hash_obj = hashlib.md5(instance_key.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        
        # Génération déterministe
        np.random.seed(hash_int % (2**31))
        embedding = np.random.normal(0, 0.1, 16).tolist()
        
        # Normalisation pour cohérence
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding
    
    def _encode_simple_key(self, key: str) -> np.ndarray:
        """Encode une clé simple (non hiérarchique)"""
        # Features basiques pour clés simples
        features = [
            len(key) / 50.0,                      # Longueur normalisée
            key.count('_') / max(len(key), 1),    # Ratio underscores
            key.count('.') / max(len(key), 1),    # Ratio points
            float(key.islower()),                 # Minuscules
            float(key.isupper()),                 # Majuscules
            float('jwt' in key.lower()),          # JWT related
            float('http' in key.lower()),         # HTTP related
            float('stream' in key.lower()),       # Stream related
            (hash(key) % 10000) / 10000.0,       # Hash unique
        ]
        
        # Compléter jusqu'à target_dim
        while len(features) < self.target_dim:
            # Ajouter des features de hash additionnelles
            hash_seed = len(features)
            np.random.seed((hash(key) + hash_seed) % (2**31))
            features.append(np.random.normal(0, 0.1))
        
        return np.array(features[:self.target_dim], dtype=np.float32)
    
    def _adjust_to_target_dim(self, vector: np.ndarray) -> np.ndarray:
        """Ajuste le vecteur à la dimension cible"""
        current_dim = len(vector)
        
        if current_dim == self.target_dim:
            return vector
        elif current_dim > self.target_dim:
            # Tronquer en gardant les features les plus importantes
            return vector[:self.target_dim]
        else:
            # Étendre avec du padding
            padding_size = self.target_dim - current_dim
            padding = np.zeros(padding_size, dtype=np.float32)
            return np.concatenate([vector, padding])
    
    def clear_cache(self):
        """Nettoie les caches"""
        self.path_analyzer.path_cache.clear()
        self.instance_embeddings_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Retourne des statistiques sur les caches"""
        return {
            'path_cache_size': len(self.path_analyzer.path_cache),
            'instance_cache_size': len(self.instance_embeddings_cache)
        }