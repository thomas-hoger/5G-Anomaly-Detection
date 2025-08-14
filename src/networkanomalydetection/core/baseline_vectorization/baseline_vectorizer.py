"""
Vectoriseur baseline : GMM + TF-IDF
"""
import networkx as nx
import numpy as np
import logging
import json
from typing import Dict, Any, Tuple, List, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Import du DimensionUnifier pour cohérence
from networkanomalydetection.core.vectorization.dimension_unifier import DimensionUnifier

logger = logging.getLogger(__name__)

class BaselineVectorizer:
    """
    Vectoriseur baseline utilisant GMM pour les nombres et TF-IDF pour le texte
    """
    
    def __init__(self, target_dim: int = 64):
        self.target_dim = target_dim
        self.gmm_model = None
        self.tfidf_model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Ajouter DimensionUnifier pour cohérence avec système intelligent
        self.dimension_unifier = DimensionUnifier(target_dim=target_dim)
        
        # IDs à traiter comme du texte (pas des nombres)
        self.id_fields = {
            "port_src", "port_dst", "id", "mnc", "mcc",
            "src_port", "dst_port", "source_port", "destination_port"
        }
        
        self.fit_stats = {}
    
    def is_numeric_value(self, value: Any, label: str = "") -> bool:
        """
        Détermine si une valeur doit être encodée avec GMM
        """
        # Vérifier si c'est un ID field
        if label.lower() in self.id_fields:
            return False
        
        # Tester si c'est un entier
        try:
            if isinstance(value, (int, float)):
                return True
            elif isinstance(value, str):
                int(value)  # Test de conversion
                return True
            return False
        except (ValueError, TypeError):
            return False
    
    def fit(self, graph: nx.MultiDiGraph) -> 'BaselineVectorizer':
        """
        Entraîne les modèles GMM et TF-IDF sur le graphe
        """
        logger.info("Entraînement du vectoriseur baseline")
        
        numeric_values = []
        text_values = []
        
        # Parcourir les nœuds
        for node_id, attrs in graph.nodes(data=True):
            value = attrs.get('label', '')
            label = str(node_id)
            
            if self.is_numeric_value(value, label):
                try:
                    numeric_values.append(float(value))
                except:
                    text_values.append(str(value))
            else:
                text_values.append(str(value))
        
        # Parcourir les arêtes
        for u, v, attrs in graph.edges(data=True):
            edge_label = attrs.get('label', '')
            text_values.append(str(edge_label))
        
        logger.info(f"Collecté {len(numeric_values)} valeurs numériques, {len(text_values)} valeurs textuelles")
        
        # Entraîner GMM
        self._fit_gmm(numeric_values)
        
        # Entraîner TF-IDF
        self._fit_tfidf(text_values)
        
        self.fit_stats = {
            "numeric_values_count": len(numeric_values),
            "text_values_count": len(text_values),
            "gmm_fitted": self.gmm_model is not None,
            "tfidf_fitted": self.tfidf_model is not None
        }
        
        self.is_fitted = True
        logger.info("Vectoriseur baseline entraîné avec succès")
        return self
    
    def _fit_gmm(self, numeric_values: List[float]):
        """Entraîne le modèle GMM"""
        if len(numeric_values) > 10:
            numeric_array = np.array(numeric_values).reshape(-1, 1)
            numeric_scaled = self.scaler.fit_transform(numeric_array)
            
            n_components = min(10, max(2, len(numeric_values) // 20))
            
            self.gmm_model = GaussianMixture(
                n_components=n_components,
                covariance_type='full',
                random_state=42
            )
            self.gmm_model.fit(numeric_scaled)
            logger.info(f"GMM entraîné avec {n_components} composants")
        else:
            logger.warning("Pas assez de données numériques pour GMM")
    
    def _fit_tfidf(self, text_values: List[str]):
        """Entraîne le modèle TF-IDF"""
        clean_text_values = [str(val).strip() for val in text_values if str(val).strip()]
        
        if len(clean_text_values) > 0:
            max_features = min(self.target_dim, len(clean_text_values))
            
            self.tfidf_model = TfidfVectorizer(
                max_features=max_features,
                lowercase=True,
                stop_words=None,
                ngram_range=(1, 2),
                min_df=1,
                token_pattern=r'(?u)\b\w+\b|[^\w\s]'
            )
            
            self.tfidf_model.fit(clean_text_values)
            logger.info(f"TF-IDF entraîné avec {len(self.tfidf_model.vocabulary_)} features")
        else:
            logger.warning("Pas de valeurs textuelles valides pour TF-IDF")
    
    def encode_value(self, value: Any, label: str = "") -> np.ndarray:
        """
        Encode une valeur avec GMM ou TF-IDF
        """
        if not self.is_fitted:
            raise ValueError("Vectoriseur non entraîné. Appelez fit() d'abord.")
        
        try:
            if self.is_numeric_value(value, label) and self.gmm_model is not None:
                return self._encode_with_gmm(value)
            else:
                return self._encode_with_tfidf(value)
        except Exception as e:
            logger.warning(f"Erreur d'encodage pour '{value}': {e}")
            return self._fallback_encoding(value)
    
    def _encode_with_gmm(self, value: Union[int, float, str]) -> np.ndarray:
        """Encode avec GMM + unification des dimensions"""
        try:
            numeric_value = float(value)
            numeric_scaled = self.scaler.transform([[numeric_value]])
            
            probabilities = self.gmm_model.predict_proba(numeric_scaled)[0]
            
            features = []
            features.extend(probabilities.tolist())
            features.append(numeric_scaled[0, 0])
            
            log_likelihood = self.gmm_model.score_samples(numeric_scaled)[0]
            features.append(log_likelihood)
            
            # Utiliser DimensionUnifier au lieu de padding manuel
            vector = np.array(features, dtype=np.float32)
            return self.dimension_unifier.unify_vector(vector, "GMM_baseline")
            
        except Exception as e:
            logger.warning(f"Erreur GMM pour '{value}': {e}")
            return self._fallback_encoding(value)
    
    def _encode_with_tfidf(self, value: Any) -> np.ndarray:
        """Encode avec TF-IDF + unification des dimensions"""
        try:
            if self.tfidf_model is None:
                return self._fallback_encoding(value)
            
            text_value = str(value).strip()
            if not text_value:
                return self._fallback_encoding(value)
            
            # Obtenir le vecteur TF-IDF
            tfidf_vector = self.tfidf_model.transform([text_value]).toarray()[0]
            
            # Utiliser DimensionUnifier au lieu de padding manuel
            return self.dimension_unifier.unify_vector(tfidf_vector, "TFIDF_baseline")
            
        except Exception as e:
            logger.warning(f"Erreur TF-IDF pour '{value}': {e}")
            return self._fallback_encoding(value)
    
    def _fallback_encoding(self, value: Any) -> np.ndarray:
        """Encodage de secours + unification des dimensions"""
        try:
            import hashlib
            
            hash_obj = hashlib.md5(str(value).encode())
            hash_int = int(hash_obj.hexdigest(), 16)
            
            np.random.seed(hash_int % (2**31))
            features = np.random.normal(0, 0.1, self.target_dim)
            
            # Utiliser DimensionUnifier pour cohérence (même si déjà bonne taille)
            return self.dimension_unifier.unify_vector(features, "FALLBACK_baseline")
            
        except Exception:
            return np.zeros(self.target_dim, dtype=np.float32)
    
    def vectorize_graph(self, graph: nx.MultiDiGraph) -> Tuple[nx.MultiDiGraph, str]:
        """
        Vectorise un graphe complet avec la méthode baseline
        """
        logger.info("Début vectorisation graphe baseline")
        
        # Entraîner le vectoriseur
        self.fit(graph)
        
        # Créer une copie du graphe pour vectorisation
        vectorized_graph = graph.copy()
        
        # Vectoriser les nœuds
        node_stats = self._vectorize_nodes(vectorized_graph)
        
        # Vectoriser les arêtes
        edge_stats = self._vectorize_edges(vectorized_graph)
        
        # Créer le rapport
        report = self._generate_report(vectorized_graph, node_stats, edge_stats)
        report_json = json.dumps(report, indent=2, default=str)
        
        logger.info("Vectorisation graphe baseline terminée")
        return vectorized_graph, report_json
    
    def _vectorize_nodes(self, graph: nx.MultiDiGraph) -> Dict[str, int]:
        """Vectorise tous les nœuds du graphe"""
        node_stats = {"gmm_encoded": 0, "tfidf_encoded": 0, "errors": 0}
        
        for node_id, attrs in graph.nodes(data=True):
            try:
                value = attrs.get('label', '')
                label = str(node_id)
                embedding = self.encode_value(value, label)
                
                graph.nodes[node_id]['embedding'] = embedding
                
                if self.is_numeric_value(value, label):
                    node_stats["gmm_encoded"] += 1
                    graph.nodes[node_id]['encoding_method'] = 'GMM'
                else:
                    node_stats["tfidf_encoded"] += 1
                    graph.nodes[node_id]['encoding_method'] = 'TF-IDF'
                    
            except Exception as e:
                logger.warning(f"Erreur encodage nœud {node_id}: {e}")
                graph.nodes[node_id]['embedding'] = np.zeros(self.target_dim, dtype=np.float32)
                graph.nodes[node_id]['encoding_method'] = 'ERROR'
                node_stats["errors"] += 1
        
        return node_stats
    
    def _vectorize_edges(self, graph: nx.MultiDiGraph) -> Dict[str, int]:
        """Vectorise toutes les arêtes du graphe"""
        edge_stats = {"tfidf_encoded": 0, "errors": 0}
        
        for u, v, attrs in graph.edges(data=True):
            try:
                edge_label = attrs.get('label', '')
                embedding = self.encode_value(edge_label)
                
                attrs['embedding'] = embedding
                attrs['encoding_method'] = 'TF-IDF'
                
                edge_stats["tfidf_encoded"] += 1
                
            except Exception as e:
                logger.warning(f"Erreur encodage arête {u}->{v}: {e}")
                attrs['embedding'] = np.zeros(self.target_dim, dtype=np.float32)
                attrs['encoding_method'] = 'ERROR'
                edge_stats["errors"] += 1
        
        return edge_stats
    
    def _generate_report(self, graph: nx.MultiDiGraph, 
                        node_stats: Dict, edge_stats: Dict) -> Dict[str, Any]:
        """Génère le rapport de vectorisation"""
        return {
            "vectorization_method": "baseline_gmm_tfidf",
            "graph_stats": {
                "total_nodes": graph.number_of_nodes(),
                "total_edges": graph.number_of_edges()
            },
            "encoding_stats": {
                "nodes": node_stats,
                "edges": edge_stats
            },
            "models_info": {
                "gmm_components": self.gmm_model.n_components if self.gmm_model else 0,
                "tfidf_features": len(self.tfidf_model.vocabulary_) if self.tfidf_model else 0,
                "target_dimension": self.target_dim
            },
            "fit_stats": self.fit_stats
        }