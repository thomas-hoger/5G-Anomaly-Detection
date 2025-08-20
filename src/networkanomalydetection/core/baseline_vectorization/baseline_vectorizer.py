"""
Vectoriseur baseline amélioré pour GNN avec Word2Vec
"""
import networkx as nx
import numpy as np
import logging
import json
import re
from typing import Dict, Any, Tuple, List, Union
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
from datetime import datetime
import ipaddress

# Import du DimensionUnifier pour cohérence
from networkanomalydetection.core.vectorization.dimension_unifier import DimensionUnifier

logger = logging.getLogger(__name__)

class BaselineVectorizer:
    """
    Vectoriseur baseline avec encodage spécialisé et Word2Vec pour GNN
    """
    
    def __init__(self, target_dim: int = 64):
        self.target_dim = target_dim
        self.gmm_model = None
        self.word2vec_model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Ajouter DimensionUnifier pour cohérence
        self.dimension_unifier = DimensionUnifier(target_dim=target_dim)
        
        # IDs à traiter comme du texte (pas des nombres)
        self.id_fields = {
            "port_src", "port_dst", "id", "mnc", "mcc"
        }
        
        self.fit_stats = {}
    
    def _detect_value_type(self, value: Any, label: str = "") -> str:
        """
        Détecte le type de valeur pour choisir l'encodage approprié
        """
        value_str = str(value).strip()
        
        # 1. Vérifier si c'est un timestamp
        if self._is_timestamp(value):
            return "timestamp"
        
        # 2. Vérifier si c'est une adresse IP
        if self._is_ip_address(value_str):
            return "ip_address"
        
        # 3. Vérifier si c'est numérique pour GMM
        if self.is_numeric_value(value, label):
            return "numeric"
        
        # 4. Sinon, c'est du texte pour Word2Vec + moyenne pondérée
        return "text"
    
    def _is_timestamp(self, value: Any) -> bool:
        """Détecte si une valeur est un timestamp"""
        try:
            num_val = float(value)
            # Timestamps Unix entre 2020 et 2030
            if 1500000000 <= num_val <= 2000000000:
                return True
            return False
        except (ValueError, TypeError):
            return False
    
    def _is_ip_address(self, value: str) -> bool:
        """Détecte si une valeur est une adresse IP"""
        try:
            ipaddress.ip_address(value)
            return True
        except:
            return False
    
    def is_numeric_value(self, value: Any, label: str = "") -> bool:
        """
        Détermine si une valeur doit être encodée avec GMM
        """
        # Vérifier si c'est un ID field
        if label.lower() in self.id_fields:
            return False
        
        # Tester si c'est un entier ET filtrer les grandes valeurs
        try:
            if isinstance(value, (int, float)):
                num_val = float(value)
            elif isinstance(value, str):
                num_val = float(value)
            else:
                return False
                
            # Filtrer les valeurs trop grandes (probable IDs/timestamps)
            if abs(num_val) > 100000:  # 100,000 = seuil
                return False
                
            return True
            
        except (ValueError, TypeError):
            return False
    
    def _normalize_hierarchical_to_compound(self, text: str) -> str:
        """
        Convertit structure hiérarchique en mots composés
        nfInstances[0].plmnList[0].mcc → nfInstances.plmnList.mcc
        """
        # Supprimer tous les indices [0], [1], etc.
        normalized = re.sub(r'\[\d+\]', '', text)
        
        # Nettoyer les doubles points qui pourraient rester
        normalized = re.sub(r'\.+', '.', normalized)
        normalized = normalized.strip('.')
        
        return normalized
    
    def _split_compound_words(self, text: str) -> List[str]:
        """
        Divise les mots composés en composants
        """
        # D'abord normaliser si c'est hiérarchique
        normalized = self._normalize_hierarchical_to_compound(text)
        
        # Séparer selon différents délimiteurs
        # Points, tirets, underscores
        components = re.split(r'[.\-_\s]+', normalized.lower())
        
        # Filtrer les composants vides
        components = [comp.strip() for comp in components if comp.strip()]
        
        return components
    
    def _calculate_word_weights(self, words: List[str]) -> List[float]:
        """
        Calcule les poids pour chaque mot (plus de poids aux derniers mots)
        """
        if not words:
            return []
        
        if len(words) == 1:
            return [1.0]
        
        weights = []
        total_weight = 0
        
        for i in range(len(words)):
            # Poids = (position + 1) / somme des positions
            weight = (i + 1)
            weights.append(weight)
            total_weight += weight
        
        # Normaliser pour que la somme = 1
        weights = [w / total_weight for w in weights]
        
        return weights
    
    def _build_word2vec_corpus(self, text_values: List[str]) -> List[List[str]]:
        """
        Construit le corpus pour entraîner Word2Vec
        """
        corpus = []
        
        for text in text_values:
            if str(text).strip():
                # Ignorer les IPs et timestamps pour le corpus Word2Vec
                if not self._is_ip_address(str(text)) and not self._is_timestamp(text):
                    words = self._split_compound_words(text)
                    if words:  # Ajouter seulement si on a des mots
                        corpus.append(words)
        
        return corpus
    
    def _train_word2vec(self, corpus: List[List[str]]):
        """
        Entraîne le modèle Word2Vec sur le corpus
        """
        if len(corpus) < 10:
            logger.warning("Corpus trop petit pour Word2Vec, utilisation d'embeddings par défaut")
            self.word2vec_model = None
            return
        
        logger.info(f"Entraînement Word2Vec sur {len(corpus)} phrases")
        
        # Paramètres optimisés pour données télécom
        self.word2vec_model = Word2Vec(
            sentences=corpus,
            vector_size=self.target_dim,
            window=5,              # Fenêtre contextuelle
            min_count=1,           # Garder même mots rares (important pour télécom)
            workers=4,
            epochs=20,             # Plus d'époques pour domaine spécialisé
            sg=1,                  # Skip-gram (meilleur pour mots rares)
            negative=10,           # Échantillonnage négatif
            alpha=0.025,           # Taux d'apprentissage
            seed=42                # Reproductibilité
        )
        
        vocab_size = len(self.word2vec_model.wv.key_to_index)
        logger.info(f"Word2Vec entraîné: {vocab_size} mots dans le vocabulaire")
        
        # Afficher quelques mots appris avec leurs voisins
        if vocab_size > 10:
            sample_words = list(self.word2vec_model.wv.key_to_index.keys())[:5]
            for word in sample_words:
                try:
                    similar = self.word2vec_model.wv.most_similar(word, topn=3)
                    logger.info(f"'{word}' similaire à: {[w for w, _ in similar]}")
                except:
                    pass
    
    def _get_word2vec_embedding(self, word: str) -> np.ndarray:
        """
        Obtient l'embedding Word2Vec d'un mot
        """
        if self.word2vec_model is None:
            # Fallback si Word2Vec n'est pas entraîné
            return self._fallback_word_embedding(word)
        
        try:
            if word in self.word2vec_model.wv:
                return self.word2vec_model.wv[word].astype(np.float32)
            else:
                # Mot inconnu : essayer de trouver des mots similaires
                similar_words = []
                for vocab_word in self.word2vec_model.wv.key_to_index:
                    if word in vocab_word or vocab_word in word:
                        similar_words.append(vocab_word)
                
                if similar_words:
                    # Prendre le premier mot similaire trouvé
                    return self.word2vec_model.wv[similar_words[0]].astype(np.float32)
                else:
                    # Aucun mot similaire : fallback
                    return self._fallback_word_embedding(word)
                    
        except Exception as e:
            logger.warning(f"Erreur Word2Vec pour '{word}': {e}")
            return self._fallback_word_embedding(word)
    
    def _fallback_word_embedding(self, word: str) -> np.ndarray:
        """
        Embedding de fallback pour mots inconnus
        """
        # Hash déterministe pour cohérence
        hash_seed = hash(word) % 10000
        np.random.seed(hash_seed)
        return np.random.normal(0, 0.1, self.target_dim).astype(np.float32)
    
    def _encode_compound_words_weighted_average(self, text: str) -> np.ndarray:
        """
        Encode les mots composés avec Word2Vec + moyenne pondérée
        """
        try:
            # 1. Diviser en mots
            words = self._split_compound_words(text)
            
            if not words:
                return self._fallback_encoding(text)
            
            # 2. Calculer les poids (plus de poids aux derniers mots)
            weights = self._calculate_word_weights(words)
            
            # 3. Obtenir l'embedding Word2Vec de chaque mot
            word_embeddings = []
            for word in words:
                word_emb = self._get_word2vec_embedding(word)
                word_embeddings.append(word_emb)
            
            # 4. Calculer la moyenne pondérée
            weighted_sum = np.zeros(self.target_dim, dtype=np.float32)
            
            for embedding, weight in zip(word_embeddings, weights):
                weighted_sum += embedding * weight
            
            return self.dimension_unifier.unify_vector(weighted_sum, "WORD2VEC_WEIGHTED")
            
        except Exception as e:
            logger.warning(f"Erreur Word2Vec pondéré pour '{text}': {e}")
            return self._fallback_encoding(text)
    
    def _encode_timestamp(self, timestamp: Any) -> np.ndarray:
        """
        Encode spécialement les timestamps
        """
        try:
            ts_float = float(timestamp)
            features = []
            
            # 1. Normaliser le timestamp (par rapport à une époque)
            # Utiliser 2024 comme référence : 1704067200 (1er Jan 2024)
            epoch_2024 = 1704067200
            ts_normalized = (ts_float - epoch_2024) / (365.25 * 24 * 3600)  # En années depuis 2024
            features.append(ts_normalized)
            
            # 2. Extraire composants temporels
            try:
                dt = datetime.fromtimestamp(ts_float)
                
                # Heure de la journée (0-1)
                hour_norm = dt.hour / 24.0
                features.append(hour_norm)
                
                # Jour de la semaine (0-1)
                weekday_norm = dt.weekday() / 7.0
                features.append(weekday_norm)
                
                # Jour du mois (0-1)
                day_norm = (dt.day - 1) / 31.0
                features.append(day_norm)
                
                # Mois de l'année (0-1)
                month_norm = (dt.month - 1) / 12.0
                features.append(month_norm)
                
                # Secondes dans la minute (pour la précision)
                seconds_norm = dt.second / 60.0
                features.append(seconds_norm)
                
                # Microsecondes (pour la haute précision)
                if hasattr(dt, 'microsecond'):
                    micro_norm = dt.microsecond / 1000000.0
                    features.append(micro_norm)
                else:
                    features.append(0.0)
                    
            except:
                # Si conversion échoue, ajouter des zéros
                features.extend([0.0] * 6)
            
            # 3. Hash du timestamp pour unicité
            ts_hash = hash(str(timestamp)) % 10000 / 10000.0
            features.append(ts_hash)
            
            # 4. Compléter avec du padding cohérent
            hash_seed = int(abs(ts_float)) % 1000
            np.random.seed(hash_seed)
            while len(features) < self.target_dim:
                features.append(np.random.normal(0, 0.01))
            
            vector = np.array(features[:self.target_dim], dtype=np.float32)
            return self.dimension_unifier.unify_vector(vector, "TIMESTAMP")
            
        except Exception as e:
            logger.warning(f"Erreur encodage timestamp '{timestamp}': {e}")
            return self._fallback_encoding(timestamp)
    
    def _encode_ip_address(self, ip: str) -> np.ndarray:
        """
        Encode spécialement les adresses IP
        """
        try:
            features = []
            
            # 1. Décoder l'adresse IP
            ip_obj = ipaddress.ip_address(ip)
            
            if isinstance(ip_obj, ipaddress.IPv4Address):
                # IPv4 : 4 octets
                octets = ip.split('.')
                for octet in octets:
                    features.append(int(octet) / 255.0)  # Normaliser 0-255 vers 0-1
                
                # Classe d'adresse
                first_octet = int(octets[0])
                class_a = 1.0 if 1 <= first_octet <= 126 else 0.0
                class_b = 1.0 if 128 <= first_octet <= 191 else 0.0
                class_c = 1.0 if 192 <= first_octet <= 223 else 0.0
                features.extend([class_a, class_b, class_c])
                
                # Réseau privé
                is_private = 1.0 if ip_obj.is_private else 0.0
                features.append(is_private)
                
                # Réseau local
                is_loopback = 1.0 if ip_obj.is_loopback else 0.0
                features.append(is_loopback)
                
                # Réseau multicast
                is_multicast = 1.0 if ip_obj.is_multicast else 0.0
                features.append(is_multicast)
                
            else:
                # IPv6 ou autre : padding avec zéros
                features.extend([0.0] * 10)
            
            # 2. Hash de l'IP complète pour unicité
            ip_hash = hash(ip) % 10000 / 10000.0
            features.append(ip_hash)
            
            # 3. Sous-réseau (approximation)
            try:
                # Pour 10.100.200.x, encoder le sous-réseau
                if '.' in ip:
                    parts = ip.split('.')
                    if len(parts) >= 3:
                        subnet_hash = hash('.'.join(parts[:3])) % 1000 / 1000.0
                        features.append(subnet_hash)
                    else:
                        features.append(0.0)
                else:
                    features.append(0.0)
            except:
                features.append(0.0)
            
            # 4. Padding
            hash_seed = hash(ip) % 1000
            np.random.seed(hash_seed)
            while len(features) < self.target_dim:
                features.append(np.random.normal(0, 0.01))
            
            vector = np.array(features[:self.target_dim], dtype=np.float32)
            return self.dimension_unifier.unify_vector(vector, "IP_ADDRESS")
            
        except Exception as e:
            logger.warning(f"Erreur encodage IP '{ip}': {e}")
            return self._fallback_encoding(ip)
    
    def fit(self, graph: nx.MultiDiGraph) -> 'BaselineVectorizer':
        """
        Entraîne les modèles GMM et Word2Vec sur le graphe
        """
        logger.info("Entraînement du vectoriseur baseline avec Word2Vec pour GNN")
        
        numeric_values = []
        text_values = []
        ip_count = 0
        timestamp_count = 0
        
        # Parcourir les nœuds
        for node_id, attrs in graph.nodes(data=True):
            value = attrs.get('label', '')
            label = str(node_id)
            
            value_type = self._detect_value_type(value, label)
            
            if value_type == "numeric":
                try:
                    numeric_values.append(float(value))
                except:
                    text_values.append(str(value))
            elif value_type == "ip_address":
                ip_count += 1
            elif value_type == "timestamp":
                timestamp_count += 1
            else:
                text_values.append(str(value))
        
        # Parcourir les arêtes
        for u, v, attrs in graph.edges(data=True):
            edge_label = attrs.get('label', '')
            value_type = self._detect_value_type(edge_label)
            
            if value_type == "ip_address":
                ip_count += 1
            elif value_type == "timestamp":
                timestamp_count += 1
            else:
                text_values.append(str(edge_label))
        
        logger.info(f"Collecté {len(numeric_values)} valeurs numériques, {ip_count} IPs, {timestamp_count} timestamps, {len(text_values)} valeurs textuelles")
        
        # Entraîner GMM
        self._fit_gmm(numeric_values)
        
        # Entraîner Word2Vec
        corpus = self._build_word2vec_corpus(text_values)
        self._train_word2vec(corpus)
        
        self.fit_stats = {
            "numeric_values_count": len(numeric_values),
            "text_values_count": len(text_values),
            "ip_addresses_count": ip_count,
            "timestamps_count": timestamp_count,
            "word2vec_vocabulary_size": len(self.word2vec_model.wv.key_to_index) if self.word2vec_model else 0,
            "corpus_size": len(corpus),
            "gmm_fitted": self.gmm_model is not None,
            "word2vec_fitted": self.word2vec_model is not None,
            "method": "baseline_word2vec_gnn"
        }
        
        self.is_fitted = True
        logger.info("Vectoriseur baseline avec Word2Vec pour GNN entraîné avec succès")
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
    
    def encode_value(self, value: Any, label: str = "") -> np.ndarray:
        """
        Encode une valeur selon son type détecté
        """
        if not self.is_fitted:
            raise ValueError("Vectoriseur non entraîné. Appelez fit() d'abord.")
        
        try:
            value_type = self._detect_value_type(value, label)
            
            if value_type == "timestamp":
                return self._encode_timestamp(value)
            elif value_type == "ip_address":
                return self._encode_ip_address(str(value))
            elif value_type == "numeric" and self.gmm_model is not None:
                return self._encode_with_gmm(value)
            else:
                return self._encode_compound_words_weighted_average(value)
                
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
        logger.info("Début vectorisation graphe baseline avec Word2Vec pour GNN")
        
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
        
        logger.info("Vectorisation graphe baseline avec Word2Vec pour GNN terminée")
        return vectorized_graph, report_json
    
    def _vectorize_nodes(self, graph: nx.MultiDiGraph) -> Dict[str, int]:
        """Vectorise tous les nœuds du graphe"""
        node_stats = {"gmm_encoded": 0, "timestamp_encoded": 0, "ip_encoded": 0, "word2vec_encoded": 0, "errors": 0}
        
        for node_id, attrs in graph.nodes(data=True):
            try:
                value = attrs.get('label', '')
                label = str(node_id)
                embedding = self.encode_value(value, label)
                
                graph.nodes[node_id]['embedding'] = embedding
                
                value_type = self._detect_value_type(value, label)
                
                if value_type == "timestamp":
                    node_stats["timestamp_encoded"] += 1
                    graph.nodes[node_id]['encoding_method'] = 'TIMESTAMP'
                elif value_type == "ip_address":
                    node_stats["ip_encoded"] += 1
                    graph.nodes[node_id]['encoding_method'] = 'IP_ADDRESS'
                elif value_type == "numeric":
                    node_stats["gmm_encoded"] += 1
                    graph.nodes[node_id]['encoding_method'] = 'GMM'
                else:
                    node_stats["word2vec_encoded"] += 1
                    graph.nodes[node_id]['encoding_method'] = 'WORD2VEC_WEIGHTED'
                    
                    # Ajouter info sur les mots décomposés
                    words = self._split_compound_words(value)
                    weights = self._calculate_word_weights(words)
                    graph.nodes[node_id]['compound_words'] = words
                    graph.nodes[node_id]['word_weights'] = weights
                    
            except Exception as e:
                logger.warning(f"Erreur encodage nœud {node_id}: {e}")
                graph.nodes[node_id]['embedding'] = np.zeros(self.target_dim, dtype=np.float32)
                graph.nodes[node_id]['encoding_method'] = 'ERROR'
                node_stats["errors"] += 1
        
        return node_stats
    
    def _vectorize_edges(self, graph: nx.MultiDiGraph) -> Dict[str, int]:
        """Vectorise toutes les arêtes du graphe"""
        edge_stats = {"timestamp_encoded": 0, "ip_encoded": 0, "word2vec_encoded": 0, "errors": 0}
        
        for u, v, attrs in graph.edges(data=True):
            try:
                edge_label = attrs.get('label', '')
                embedding = self.encode_value(edge_label)
                
                attrs['embedding'] = embedding
                
                value_type = self._detect_value_type(edge_label)
                
                if value_type == "timestamp":
                    edge_stats["timestamp_encoded"] += 1
                    attrs['encoding_method'] = 'TIMESTAMP'
                elif value_type == "ip_address":
                    edge_stats["ip_encoded"] += 1
                    attrs['encoding_method'] = 'IP_ADDRESS'
                else:
                    edge_stats["word2vec_encoded"] += 1
                    attrs['encoding_method'] = 'WORD2VEC_WEIGHTED'
                    
                    # Ajouter info sur les mots décomposés
                    words = self._split_compound_words(edge_label)
                    weights = self._calculate_word_weights(words)
                    attrs['compound_words'] = words
                    attrs['word_weights'] = weights
                
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
            "vectorization_method": "baseline_word2vec_gnn_optimized",
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
                "word2vec_vocabulary_size": len(self.word2vec_model.wv.key_to_index) if self.word2vec_model else 0,
                "target_dimension": self.target_dim
            },
            "fit_stats": self.fit_stats,
            "gnn_improvements": [
                "Word2Vec pour similarité sémantique des mots",
                "Moyenne pondérée préservant hiérarchie",
                "Encodage spécialisé timestamps et IPs",
                "Gestion intelligente mots inconnus",
                "Optimisé pour propagation de messages GNN"
            ]
        }