"""
Orchestrateur principal pour la vectorisation intelligente de graphes
"""
import networkx as nx
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from collections import defaultdict, Counter

from .entity_classifier import MultiLevelEntityClassifier, EdgeClassifier
from .specialized_encoders import EncoderFactory
from .dimension_unifier import DimensionUnifier
from .hierarchical_handler import HierarchicalEdgeEncoder

logger = logging.getLogger(__name__)

class VectorizationReport: 
    """Rapport de vectorisation"""
    
    def __init__(self):
        self.node_classifications = Counter()
        self.edge_classifications = Counter()
        self.encoding_methods = Counter()
        self.dimension_adjustments = Counter()
        self.errors = []
        self.processing_time = 0.0
        self.total_nodes = 0
        self.total_edges = 0

class IntelligentGraphVectorizer:
    """Vectoriseur intelligent pour graphes NetworkX"""
    
    def __init__(self, node_dim: int = 64, edge_dim: int = 64):
        """
        Initialise le vectoriseur
        
        Args:
            node_dim: Dimension des vecteurs de nœuds
            edge_dim: Dimension des vecteurs d'arêtes
        """
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        # Composants principaux
        self.node_classifier = MultiLevelEntityClassifier()
        self.edge_classifier = EdgeClassifier()
        self.node_encoder_factory = EncoderFactory(target_dim=node_dim)
        self.edge_encoder_factory = EncoderFactory(target_dim=edge_dim)
        self.hierarchical_encoder = HierarchicalEdgeEncoder(target_dim=edge_dim)
        self.dimension_unifier = DimensionUnifier(target_dim=max(node_dim, edge_dim))
        
        # État d'initialisation
        self.is_fitted = False
        self.node_stats = {}
        self.edge_stats = {}
    
    def fit(self, graph: nx.MultiDiGraph) -> 'IntelligentGraphVectorizer':
        """
        Analyse le graphe et initialise les encodeurs
        
        Args:
            graph: Graphe à analyser
            
        Returns:
            Self pour chaînage
        """
        logger.info(f"Fitting vectorizer on graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        try:
            # Collecter toutes les valeurs
            all_node_labels = [str(attrs.get('label', '')) for _, attrs in graph.nodes(data=True)]
            all_edge_labels = [str(attrs.get('label', '')) for _, _, attrs in graph.edges(data=True)]
            
            # Filtrer les valeurs vides
            clean_node_labels = [label for label in all_node_labels if label.strip()]
            clean_edge_labels = [label for label in all_edge_labels if label.strip()]
            
            logger.info(f"Collected {len(clean_node_labels)} node labels, {len(clean_edge_labels)} edge labels")
            
            # Entraîner les encodeurs universels
            if clean_node_labels:
                self.node_encoder_factory.fit_universal(clean_node_labels)
            if clean_edge_labels:
                self.edge_encoder_factory.fit_universal(clean_edge_labels)
            
            # Analyser les statistiques
            self._analyze_graph_statistics(graph)
            
            self.is_fitted = True
            logger.info("Vectorizer fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting vectorizer: {e}")
            raise
        
        return self
    
    def transform(self, graph: nx.MultiDiGraph) -> Tuple[nx.MultiDiGraph, VectorizationReport]:
        """
        Transforme le graphe en version vectorisée
        
        Args:
            graph: Graphe à transformer
            
        Returns:
            Tuple[nx.MultiDiGraph, VectorizationReport]: Graphe vectorisé et rapport
        """
        if not self.is_fitted:
            logger.warning("Vectorizer not fitted, calling fit() first")
            self.fit(graph)
        
        logger.info(f"Transforming graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Créer une copie du graphe
        vectorized_graph = graph.copy()
        report = VectorizationReport()
        report.total_nodes = graph.number_of_nodes()
        report.total_edges = graph.number_of_edges()
        
        try:
            # Vectoriser les nœuds
            self._vectorize_nodes(vectorized_graph, report)
            
            # Vectoriser les arêtes
            self._vectorize_edges(vectorized_graph, report)
            
            # Validation finale
            self._validate_vectorized_graph(vectorized_graph, report)
            
            logger.info(f"Graph vectorization completed successfully")
            
        except Exception as e:
            logger.error(f"Error transforming graph: {e}")
            report.errors.append(f"Transform error: {str(e)}")
            raise
        
        return vectorized_graph, report
    
    def fit_transform(self, graph: nx.MultiDiGraph) -> Tuple[nx.MultiDiGraph, VectorizationReport]:
        """
        Fit et transform en une seule opération
        
        Args:
            graph: Graphe à traiter
            
        Returns:
            Tuple[nx.MultiDiGraph, VectorizationReport]: Graphe vectorisé et rapport
        """
        return self.fit(graph).transform(graph)
    
    def _vectorize_nodes(self, graph: nx.MultiDiGraph, report: VectorizationReport):
        """Vectorise tous les nœuds du graphe"""
        logger.info("Vectorizing nodes...")
        
        for node_id, attrs in graph.nodes(data=True):
            try:
                # Récupérer la valeur du nœud
                node_value = attrs.get('label', '')
                node_type = attrs.get('node_type', 0)
                
                # Classifier l'entité
                classification = self.node_classifier.classify_entity(node_value)
                report.node_classifications[classification.entity_type] += 1
                
                # Encoder avec l'encodeur approprié
                if classification.entity_type == 'TEXT' and not str(node_value).strip():
                    # Nœud central vide - encodage spécial
                    embedding = self._encode_central_node(node_id, attrs, graph)
                    report.encoding_methods['CENTRAL_NODE'] += 1
                else:
                    # Encodage spécialisé ou universel
                    embedding = self.node_encoder_factory.encode_value(
                        node_value, 
                        classification.entity_type,
                        classification.confidence,
                        classification.metadata
                    )
                    report.encoding_methods[classification.entity_type] += 1
                
                # Ajouter des features communes
                common_features = self._extract_common_node_features(node_id, attrs, graph)
                
                # Combiner embedding spécialisé + features communes
                specialized_dim = len(embedding)
                common_dim = self.node_dim - specialized_dim
                
                if common_dim > 0:
                    common_features_adjusted = common_features[:common_dim]
                    if len(common_features_adjusted) < common_dim:
                        padding = np.zeros(common_dim - len(common_features_adjusted))
                        common_features_adjusted = np.concatenate([common_features_adjusted, padding])
                    
                    final_embedding = np.concatenate([embedding, common_features_adjusted])
                else:
                    final_embedding = embedding
                
                # Unifier la dimension
                final_embedding = self.dimension_unifier.unify_vector(
                    final_embedding, f"node_{classification.entity_type}"
                )
                
                # Stocker l'embedding dans le graphe
                graph.nodes[node_id]['embedding'] = final_embedding
                graph.nodes[node_id]['entity_type'] = classification.entity_type
                graph.nodes[node_id]['classification_confidence'] = classification.confidence
                
            except Exception as e:
                logger.warning(f"Error vectorizing node {node_id}: {e}")
                report.errors.append(f"Node {node_id}: {str(e)}")
                # Embedding par défaut en cas d'erreur
                graph.nodes[node_id]['embedding'] = np.zeros(self.node_dim, dtype=np.float32)
                graph.nodes[node_id]['entity_type'] = 'ERROR'
                graph.nodes[node_id]['classification_confidence'] = 0.0
    
    def _vectorize_edges(self, graph: nx.MultiDiGraph, report: VectorizationReport):
        """Vectorise toutes les arêtes du graphe - VERSION CORRIGÉE"""
        logger.info("Vectorizing edges...")
        
        #  CORRECTION : Utiliser edges(data=True, keys=True) pour MultiDiGraph
        for u, v, key, attrs in graph.edges(data=True, keys=True):
            try:
                # Récupérer la clé d'arête
                edge_key = attrs.get('label', '')
                
                # Classifier l'arête
                edge_classification = self.edge_classifier.classify_edge(edge_key)
                report.edge_classifications[edge_classification.entity_type] += 1
                
                # Encoder l'arête
                if '[' in edge_key and ']' in edge_key:
                    # Arête hiérarchique
                    edge_embedding = self.hierarchical_encoder.encode_hierarchical_edge(edge_key)
                    report.encoding_methods['HIERARCHICAL_EDGE'] += 1
                else:
                    # Arête simple
                    edge_embedding = self.edge_encoder_factory.encode_value(
                        edge_key,
                        edge_classification.entity_type,
                        edge_classification.confidence,
                        edge_classification.metadata
                    )
                    report.encoding_methods[edge_classification.entity_type] += 1
                
                # Ajouter le contexte topologique
                context_features = self._extract_edge_context_features(u, v, graph)
                
                # Combiner embedding + contexte
                embedding_dim = len(edge_embedding)
                context_dim = self.edge_dim - embedding_dim
                
                if context_dim > 0:
                    context_features_adjusted = context_features[:context_dim]
                    if len(context_features_adjusted) < context_dim:
                        padding = np.zeros(context_dim - len(context_features_adjusted))
                        context_features_adjusted = np.concatenate([context_features_adjusted, padding])
                    
                    final_embedding = np.concatenate([edge_embedding, context_features_adjusted])
                else:
                    final_embedding = edge_embedding
                
                # Unifier la dimension
                final_embedding = self.dimension_unifier.unify_vector(
                    final_embedding, f"edge_{edge_classification.entity_type}"
                )
                
                #  CORRECTION : Stocker avec la clé pour MultiDiGraph
                graph.edges[u, v, key]['embedding'] = final_embedding
                graph.edges[u, v, key]['entity_type'] = edge_classification.entity_type
                graph.edges[u, v, key]['classification_confidence'] = edge_classification.confidence
                
            except Exception as e:
                logger.warning(f"Error vectorizing edge {u}->{v} (key={key}): {e}")
                report.errors.append(f"Edge {u}->{v} (key={key}): {str(e)}")
                #  CORRECTION : Embedding par défaut avec clé
                graph.edges[u, v, key]['embedding'] = np.zeros(self.edge_dim, dtype=np.float32)
                graph.edges[u, v, key]['entity_type'] = 'ERROR'
                graph.edges[u, v, key]['classification_confidence'] = 0.0
    
    def _encode_central_node(self, node_id: int, attrs: Dict, graph: nx.MultiDiGraph) -> np.ndarray:
        """Encodage spécial pour nœuds centraux (souvent vides)"""
        # Features structurelles pour nœuds centraux
        features = [
            # Position dans le graphe
            graph.degree(node_id) / 100.0,                    # Degré normalisé
            graph.in_degree(node_id) / 100.0,                 # Degré entrant
            graph.out_degree(node_id) / 100.0,                # Degré sortant
            
            # Métadonnées
            attrs.get('node_type', 0) / 2.0,                  # Type normalisé
            attrs.get('packet_id', 0) / 1000.0,               # ID paquet normalisé
            
            # Features topologiques
            len(list(graph.neighbors(node_id))) / 50.0,       # Nombre voisins
            float(attrs.get('node_type', 0) == 1),            # Est central
            
            # Hash unique pour différenciation
            (hash(f"central_{node_id}") % 10000) / 10000.0,
        ]
        
        # Compléter avec des zéros si nécessaire
        while len(features) < self.node_dim // 2:
            features.append(0.0)
        
        return np.array(features[:self.node_dim // 2], dtype=np.float32)
    
    def _extract_common_node_features(self, node_id: int, attrs: Dict, 
                                    graph: nx.MultiDiGraph) -> np.ndarray:
        """Extrait des features communes pour tous les nœuds"""
        
        node_value = str(attrs.get('label', ''))
        
        # Features textuelles basiques
        text_features = [
            len(node_value) / 100.0,                          # Longueur
            node_value.count('.') / max(len(node_value), 1),  # Densité points
            node_value.count('-') / max(len(node_value), 1),  # Densité tirets
            node_value.count('[') / max(len(node_value), 1),  # Densité crochets
            float(node_value.isdigit()),                      # Est numérique
            float(node_value.isalpha()),                      # Est alphabétique
        ]
        
        # Features de graphe
        graph_features = [
            graph.degree(node_id) / 100.0,                   # Degré normalisé
            graph.in_degree(node_id) / 100.0,                # Degré entrant normalisé
            graph.out_degree(node_id) / 100.0,               # Degré sortant normalisé
            len(list(graph.neighbors(node_id))) / 50.0,      # Voisins normalisé
        ]
        
        # Features de métadonnées
        meta_features = [
            attrs.get('node_type', 0) / 2.0,                 # Type normalisé
            attrs.get('packet_id', 0) / 1000.0,              # Packet ID normalisé
            float(attrs.get('node_type', 0) == 1),           # Est central
            float(attrs.get('node_type', 0) == 2),           # Est paramètre
        ]
        
        # Combiner toutes les features
        all_features = text_features + graph_features + meta_features
        
        return np.array(all_features, dtype=np.float32)
    
    def _extract_edge_context_features(self, source: int, target: int, 
                                     graph: nx.MultiDiGraph) -> np.ndarray:
        """Extrait des features contextuelles pour les arêtes"""
        
        try:
            # Features des nœuds source et target
            source_attrs = graph.nodes[source]
            target_attrs = graph.nodes[target]
            
            context_features = [
                # Types des nœuds
                source_attrs.get('node_type', 0) / 2.0,              # Type source
                target_attrs.get('node_type', 0) / 2.0,              # Type target
                float(source_attrs.get('node_type', 0) == 1),        # Source est central
                float(target_attrs.get('node_type', 0) == 2),        # Target est paramètre
                
                # Degrés des nœuds
                graph.degree(source) / 100.0,                        # Degré source
                graph.degree(target) / 100.0,                        # Degré target
                
                # Features topologiques
                float(graph.degree(source) > graph.degree(target)),  # Source plus connecté
                abs(graph.degree(source) - graph.degree(target)) / 100.0, # Différence degrés
                
                # Features temporelles (si disponible)
                abs(source_attrs.get('packet_id', 0) - target_attrs.get('packet_id', 0)) / 1000.0,
                
                # Features d'unicité
                (hash(f"{source}_{target}") % 10000) / 10000.0,      # Hash unique de l'arête
            ]
            
            return np.array(context_features, dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error extracting edge context for {source}->{target}: {e}")
            return np.zeros(10, dtype=np.float32)
    
    def _analyze_graph_statistics(self, graph: nx.MultiDiGraph):
        """Analyse les statistiques du graphe"""
        
        # Statistiques des nœuds
        node_types = Counter()
        node_labels_length = []
        
        for _, attrs in graph.nodes(data=True):
            node_types[attrs.get('node_type', 'unknown')] += 1
            label = str(attrs.get('label', ''))
            node_labels_length.append(len(label))
        
        self.node_stats = {
            'types': dict(node_types),
            'avg_label_length': np.mean(node_labels_length) if node_labels_length else 0,
            'max_label_length': max(node_labels_length) if node_labels_length else 0,
        }
        
        # Statistiques des arêtes
        edge_labels = []
        hierarchical_edges = 0
        
        for _, _, attrs in graph.edges(data=True):
            edge_label = str(attrs.get('label', ''))
            edge_labels.append(edge_label)
            if '[' in edge_label and ']' in edge_label:
                hierarchical_edges += 1
        
        self.edge_stats = {
            'total': len(edge_labels),
            'hierarchical_count': hierarchical_edges,
            'hierarchical_ratio': hierarchical_edges / len(edge_labels) if edge_labels else 0,
            'unique_labels': len(set(edge_labels)),
        }
        
        logger.info(f"Graph statistics: {self.node_stats['types']} node types, "
                   f"{self.edge_stats['hierarchical_count']} hierarchical edges")
    
    def _validate_vectorized_graph(self, graph: nx.MultiDiGraph, report: VectorizationReport):
        """Valide que la vectorisation est correcte - VERSION CORRIGÉE"""
        
        # Vérifier que tous les nœuds ont des embeddings
        nodes_without_embedding = []
        for node_id, attrs in graph.nodes(data=True):
            if 'embedding' not in attrs:
                nodes_without_embedding.append(node_id)
            elif len(attrs['embedding']) != self.node_dim:
                report.errors.append(f"Node {node_id} has wrong embedding dimension: {len(attrs['embedding'])}")
        
        if nodes_without_embedding:
            report.errors.append(f"Nodes without embedding: {nodes_without_embedding}")
        
        #  CORRECTION : Vérifier les arêtes avec clés pour MultiDiGraph
        edges_without_embedding = []
        for u, v, key, attrs in graph.edges(data=True, keys=True):
            if 'embedding' not in attrs:
                edges_without_embedding.append((u, v, key))
            elif len(attrs['embedding']) != self.edge_dim:
                report.errors.append(f"Edge {u}->{v} (key={key}) has wrong embedding dimension: {len(attrs['embedding'])}")
        
        if edges_without_embedding:
            report.errors.append(f"Edges without embedding: {edges_without_embedding}")
        
        # Statistiques finales
        if graph.number_of_nodes() > 0:
            embeddings_sample = [attrs['embedding'] for _, attrs in list(graph.nodes(data=True))[:100]]
            if embeddings_sample:
                embeddings_array = np.array(embeddings_sample)
                report.dimension_adjustments['node_mean'] = np.mean(embeddings_array)
                report.dimension_adjustments['node_std'] = np.std(embeddings_array)
                report.dimension_adjustments['node_sparsity'] = np.mean(embeddings_array == 0)
        
        logger.info(f"Validation completed. Errors: {len(report.errors)}")
    
    def get_vectorization_summary(self, report: VectorizationReport) -> Dict[str, Any]:
        """Retourne un résumé de la vectorisation"""
        
        return {
            'total_nodes': report.total_nodes,
            'total_edges': report.total_edges,
            'node_classifications': dict(report.node_classifications),
            'edge_classifications': dict(report.edge_classifications),
            'encoding_methods': dict(report.encoding_methods),
            'errors_count': len(report.errors),
            'errors': report.errors[:10],  # Premiers 10 erreurs seulement
            'node_stats': self.node_stats,
            'edge_stats': self.edge_stats,
            'dimensions': {
                'node_dim': self.node_dim,
                'edge_dim': self.edge_dim
            },
            'quality_metrics': {
                'node_success_rate': ((report.total_nodes - len([e for e in report.errors if 'Node' in e])) / max(report.total_nodes, 1)) * 100,
                'edge_success_rate': ((report.total_edges - len([e for e in report.errors if 'Edge' in e])) / max(report.total_edges, 1)) * 100,
                'overall_success_rate': ((report.total_nodes + report.total_edges - len(report.errors)) / max(report.total_nodes + report.total_edges, 1)) * 100
            }
        }
    
    def save_vectorized_graph(self, graph: nx.MultiDiGraph, filepath: str):
        """Sauvegarde le graphe vectorisé"""
        try:
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(graph, f)
            logger.info(f"Vectorized graph saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving vectorized graph: {e}")
            raise
    
    def load_vectorized_graph(self, filepath: str) -> nx.MultiDiGraph:
        """Charge un graphe vectorisé"""
        try:
            import pickle
            with open(filepath, 'rb') as f:
                graph = pickle.load(f)
            logger.info(f"Vectorized graph loaded from {filepath}")
            return graph
        except Exception as e:
            logger.error(f"Error loading vectorized graph: {e}")
            raise
    
    def get_embeddings_matrices(self, graph: nx.MultiDiGraph) -> Dict[str, np.ndarray]:
        """
        Extrait les matrices d'embeddings du graphe vectorisé - VERSION CORRIGÉE
        
        Args:
            graph: Graphe vectorisé
            
        Returns:
            Dict contenant les matrices d'embeddings
        """
        try:
            # Extraire les embeddings des nœuds
            node_embeddings = []
            node_ids = []
            for node_id, attrs in graph.nodes(data=True):
                if 'embedding' in attrs:
                    node_embeddings.append(attrs['embedding'])
                    node_ids.append(node_id)
            
            #  CORRECTION : Extraire les embeddings des arêtes avec clés
            edge_embeddings = []
            edge_indices = []
            for u, v, key, attrs in graph.edges(data=True, keys=True):
                if 'embedding' in attrs:
                    edge_embeddings.append(attrs['embedding'])
                    # Mapper les IDs de nœuds aux indices
                    u_idx = node_ids.index(u) if u in node_ids else -1
                    v_idx = node_ids.index(v) if v in node_ids else -1
                    if u_idx >= 0 and v_idx >= 0:
                        edge_indices.append([u_idx, v_idx])
            
            return {
                'node_embeddings': np.array(node_embeddings) if node_embeddings else np.empty((0, self.node_dim)),
                'edge_embeddings': np.array(edge_embeddings) if edge_embeddings else np.empty((0, self.edge_dim)),
                'edge_index': np.array(edge_indices).T if edge_indices else np.empty((2, 0), dtype=int),
                'node_ids': node_ids,
                'num_nodes': len(node_ids),
                'num_edges': len(edge_embeddings)
            }
            
        except Exception as e:
            logger.error(f"Error extracting embedding matrices: {e}")
            raise
    
    def validate_embeddings_quality(self, graph: nx.MultiDiGraph) -> Dict[str, Any]:
        """
        Valide la qualité des embeddings générés
        
        Args:
            graph: Graphe vectorisé
            
        Returns:
            Dict: Métriques de qualité
        """
        try:
            matrices = self.get_embeddings_matrices(graph)
            
            node_embeddings = matrices['node_embeddings']
            edge_embeddings = matrices['edge_embeddings']
            
            quality_metrics = {}
            
            if len(node_embeddings) > 0:
                quality_metrics['node_metrics'] = {
                    'mean': float(np.mean(node_embeddings)),
                    'std': float(np.std(node_embeddings)),
                    'min': float(np.min(node_embeddings)),
                    'max': float(np.max(node_embeddings)),
                    'sparsity': float(np.mean(node_embeddings == 0)),
                    'dimension': node_embeddings.shape[1],
                    'count': node_embeddings.shape[0]
                }
            
            if len(edge_embeddings) > 0:
                quality_metrics['edge_metrics'] = {
                    'mean': float(np.mean(edge_embeddings)),
                    'std': float(np.std(edge_embeddings)),
                    'min': float(np.min(edge_embeddings)),
                    'max': float(np.max(edge_embeddings)),
                    'sparsity': float(np.mean(edge_embeddings == 0)),
                    'dimension': edge_embeddings.shape[1],
                    'count': edge_embeddings.shape[0]
                }
            
            # Vérifications de cohérence
            quality_metrics['consistency_checks'] = {
                'all_nodes_have_embeddings': all('embedding' in attrs for _, attrs in graph.nodes(data=True)),
                'all_edges_have_embeddings': all('embedding' in attrs for _, _, attrs in graph.edges(data=True)),
                'node_dimension_consistent': len(set(len(attrs['embedding']) for _, attrs in graph.nodes(data=True) if 'embedding' in attrs)) <= 1,
                'edge_dimension_consistent': len(set(len(attrs['embedding']) for _, _, attrs in graph.edges(data=True) if 'embedding' in attrs)) <= 1
            }
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error validating embeddings quality: {e}")
            return {'error': str(e)}
    
    def clear_caches(self):
        """Nettoie tous les caches"""
        try:
            self.hierarchical_encoder.clear_cache()
            logger.info("Caches cleared successfully")
        except Exception as e:
            logger.warning(f"Error clearing caches: {e}")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Retourne l'usage mémoire approximatif"""
        try:
            cache_stats = self.hierarchical_encoder.get_cache_stats()
            
            return {
                'hierarchical_caches': cache_stats,
                'encoder_count': len(self.node_encoder_factory.encoders) + len(self.edge_encoder_factory.encoders),
                'fitted_universal_encoders': {
                    'node_encoder': self.node_encoder_factory.universal_encoder.fitted,
                    'edge_encoder': self.edge_encoder_factory.universal_encoder.fitted
                }
            }
            
        except Exception as e:
            logger.warning(f"Error getting memory usage: {e}")
            return {'error': str(e)}