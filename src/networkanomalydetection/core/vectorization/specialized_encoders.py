"""
Encodeurs spécialisés pour différents types d'entités réseau 5G
"""
import numpy as np
import hashlib
import math
from datetime import datetime
from typing import Any, List, Dict, Union
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logger = logging.getLogger(__name__)

class BaseEncoder:
    """Classe de base pour tous les encodeurs"""
    
    def __init__(self, output_dim: int = 32):
        self.output_dim = output_dim
    
    def encode(self, value: Any, metadata: Dict = None) -> np.ndarray:
        """Encode une valeur en vecteur"""
        raise NotImplementedError
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalise un vecteur"""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def _stable_hash_features(self, value: str, num_features: int) -> np.ndarray:
        """Génère des features basées sur hash stable"""
        hash_obj = hashlib.md5(str(value).encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        
        # Génération déterministe
        np.random.seed(hash_int % (2**31))
        features = np.random.normal(0, 0.1, num_features)
        
        return features

class IPAddressEncoder(BaseEncoder):
    """Encodeur spécialisé pour adresses IP"""
    
    def encode(self, ip: str, metadata: Dict = None) -> np.ndarray:
        """Encode une adresse IP"""
        try:
            octets = [int(x) for x in ip.split('.')]
            
            features = [
                # Octets normalisés
                octets[0] / 255.0,
                octets[1] / 255.0, 
                octets[2] / 255.0,
                octets[3] / 255.0,
                
                # Features réseau
                float(ip.startswith('10.')),           # Réseau privé classe A
                float(ip.startswith('192.168.')),      # Réseau privé classe C
                float(ip.startswith('172.')),          # Réseau privé classe B
                float(ip.startswith('127.')),          # Loopback
                
                # Features de classe réseau
                octets[0] / 255.0 * 0.8,              # Poids classe réseau
                (octets[0] + octets[1]) / 510.0,      # Sous-réseau
                
                # Features de position
                float(ip.endswith('.1')),             # Probable gateway
                float(ip.endswith('.255')),           # Probable broadcast
                
                # Features statistiques
                np.mean(octets) / 255.0,              # Moyenne octets
                np.std(octets) / 255.0,               # Écart-type octets
                
                # Hash unique pour différenciation
                (hash(ip) % 10000) / 10000.0
            ]
            
            # Compléter jusqu'à output_dim si nécessaire
            while len(features) < self.output_dim:
                features.extend(self._stable_hash_features(ip, min(5, self.output_dim - len(features))))
                features = features[:self.output_dim]
            
            return np.array(features[:self.output_dim], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error encoding IP {ip}: {e}")
            return self._stable_hash_features(ip, self.output_dim)

class TimestampEncoder(BaseEncoder):
    """Encodeur spécialisé pour timestamps"""
    
    def encode(self, timestamp: Union[int, float], metadata: Dict = None) -> np.ndarray:
        """Encode un timestamp"""
        try:
            dt = datetime.fromtimestamp(timestamp)
            
            features = [
                # Features temporelles cycliques
                math.sin(2 * math.pi * dt.hour / 24),           # Heure (cyclique)
                math.cos(2 * math.pi * dt.hour / 24),
                math.sin(2 * math.pi * dt.weekday() / 7),       # Jour semaine (cyclique)
                math.cos(2 * math.pi * dt.weekday() / 7),
                math.sin(2 * math.pi * dt.day / 31),            # Jour mois (cyclique)
                math.cos(2 * math.pi * dt.day / 31),
                
                # Features temporelles linéaires normalisées
                dt.hour / 24.0,                                 # Heure linéaire
                dt.weekday() / 7.0,                            # Jour semaine linéaire
                dt.day / 31.0,                                 # Jour mois linéaire
                dt.month / 12.0,                               # Mois
                
                # Features de précision
                (timestamp % 1),                               # Partie décimale (millisecondes)
                (timestamp % 60) / 60.0,                       # Secondes
                (timestamp % 3600) / 3600.0,                   # Minutes dans l'heure
                
                # Features contextuelles
                float(9 <= dt.hour <= 17),                     # Heures bureau
                float(dt.weekday() < 5),                       # Jour ouvrable
                float(dt.hour < 6 or dt.hour > 22),           # Heures creuses
            ]
            
            # Compléter si nécessaire
            while len(features) < self.output_dim:
                features.append((timestamp % (10**(len(features)+1))) / (10**(len(features)+1)))
            
            return np.array(features[:self.output_dim], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error encoding timestamp {timestamp}: {e}")
            return self._stable_hash_features(str(timestamp), self.output_dim)

class UUIDEncoder(BaseEncoder):
    """Encodeur spécialisé pour UUIDs"""
    
    def encode(self, uuid: str, metadata: Dict = None) -> np.ndarray:
        """Encode un UUID"""
        try:
            # Retirer les tirets
            hex_part = uuid.replace('-', '').lower()
            
            features = [
                # Features de structure
                len(hex_part) / 32.0,                          # Longueur normalisée
                
                # Ratio des types de caractères
                sum(c in '0123456789' for c in hex_part) / len(hex_part),  # Ratio chiffres
                sum(c in 'abcdef' for c in hex_part) / len(hex_part),      # Ratio lettres
                
                # Features de distribution
                int(hex_part[0], 16) / 16.0,                   # Premier hex digit
                int(hex_part[-1], 16) / 16.0,                  # Dernier hex digit
                int(hex_part[len(hex_part)//2], 16) / 16.0,    # Milieu
                
                # Features entropiques
                len(set(hex_part)) / 16.0,                     # Diversité caractères
                
                # Version UUID (si détectable)
                int(uuid.split('-')[2][0], 16) / 16.0,         # Version field
            ]
            
            # Ajouter des features basées sur segments UUID
            segments = uuid.split('-')
            for i, segment in enumerate(segments[:4]):  # Limiter à 4 segments
                if i < len(segments):
                    segment_value = int(segment, 16) if segment else 0
                    features.append((segment_value % 10000) / 10000.0)
            
            # Compléter avec hash si nécessaire
            while len(features) < self.output_dim:
                remaining = self.output_dim - len(features)
                features.extend(self._stable_hash_features(uuid, min(remaining, 10)))
                features = features[:self.output_dim]
            
            return np.array(features[:self.output_dim], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error encoding UUID {uuid}: {e}")
            return self._stable_hash_features(uuid, self.output_dim)

class Service5GEncoder(BaseEncoder):
    """Encodeur spécialisé pour services 5G"""
    
    def __init__(self, output_dim: int = 32):
        super().__init__(output_dim)
        
        # Embeddings des Network Functions
        self.nf_embeddings = {
            'udm': [1, 0, 0, 0, 0, 0.9],      # User Data Management
            'amf': [0, 1, 0, 0, 0, 0.9],      # Access Management
            'smf': [0, 0, 1, 0, 0, 0.8],      # Session Management
            'pcf': [0, 0, 0, 1, 0, 0.8],      # Policy Control
            'ausf': [0, 0, 0, 0, 1, 0.8],     # Authentication Server
            'nrf': [1, 1, 0, 0, 0, 1.0],      # NF Repository
            'udr': [1, 0, 1, 0, 0, 0.7],      # Unified Data Repository
        }
        
        # Embeddings des types de services
        self.service_embeddings = {
            'sdm': [1, 0, 0, 0.9],            # Subscriber Data Management
            'uecm': [0, 1, 0, 0.7],           # UE Context Management
            'ueau': [0, 0, 1, 0.8],           # UE Authentication
            'ee': [1, 1, 0, 0.6],             # Event Exposure
            'disc': [0, 1, 1, 0.8],           # Discovery
            'auth': [0, 0, 1, 1.0],           # Authentication
        }
    
    def encode(self, service_name: str, metadata: Dict = None) -> np.ndarray:
        """Encode un nom de service 5G"""
        try:
            # Analyser la structure du service
            parts = service_name.lower().split('-')
            
            # Extraire NF et service type
            nf_part = parts[0][1:] if parts[0].startswith('n') else parts[0]
            service_parts = parts[1:] if len(parts) > 1 else []
            
            # Encoding NF
            nf_vector = self.nf_embeddings.get(nf_part, [0, 0, 0, 0, 0, 0.1])
            
            # Encoding service type (combinaison des parties)
            service_vector = [0, 0, 0, 0.1]
            for part in service_parts:
                if part in self.service_embeddings:
                    part_vector = self.service_embeddings[part]
                    service_vector = [a + b for a, b in zip(service_vector, part_vector)]
            
            # Normaliser service_vector
            service_sum = sum(service_vector)
            if service_sum > 0:
                service_vector = [v / service_sum for v in service_vector]
            
            # Features structurelles
            structural_features = [
                len(service_name) / 50.0,                     # Longueur normalisée
                len(parts),                                   # Nombre de composants
                service_name.count('-') / 5.0,                # Complexité
                float('policy' in service_name),              # Politique
                float('control' in service_name),             # Contrôle
                float('data' in service_name),                # Données
                float('auth' in service_name),                # Authentification
            ]
            
            # Combiner tous les vecteurs
            features = nf_vector + service_vector + structural_features
            
            # Hash pour unicité
            hash_features = self._stable_hash_features(service_name, 
                                                     max(1, self.output_dim - len(features)))
            features.extend(hash_features)
            
            return np.array(features[:self.output_dim], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error encoding service {service_name}: {e}")
            return self._stable_hash_features(service_name, self.output_dim)

class UniversalTextEncoder:
    """Encodeur universel pour texte non-spécialisé"""
    
    def __init__(self, target_dim: int = 32, max_features: int = 1000):
        self.target_dim = target_dim
        self.max_features = min(max_features, target_dim)
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            lowercase=True,
            stop_words=None,
            ngram_range=(1, 2)
        )
        self.fitted = False
    
    def fit(self, texts: list):
        """Entraîne l'encodeur sur un corpus de textes"""
        try:
            clean_texts = [str(text).strip() for text in texts if str(text).strip()]
            if clean_texts:
                self.vectorizer.fit(clean_texts)
                self.fitted = True
                logger.info(f"Universal encoder fitted on {len(clean_texts)} texts")
            else:
                logger.warning("No valid texts to fit universal encoder")
        except Exception as e:
            logger.error(f"Error fitting universal encoder: {e}")
            self.fitted = False
    
    def encode(self, text: str) -> np.ndarray:
        """Encode un texte en vecteur TF-IDF"""
        try:
            if not self.fitted:
                return self._hash_encode(text)
            
            tfidf_vector = self.vectorizer.transform([str(text)]).toarray()[0]
            
            if len(tfidf_vector) < self.target_dim:
                padding_size = self.target_dim - len(tfidf_vector)
                hash_features = self._hash_encode(text, padding_size)
                full_vector = np.concatenate([tfidf_vector, hash_features])
            else:
                full_vector = tfidf_vector[:self.target_dim]
            
            return full_vector.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Error in universal encoding for '{text}': {e}")
            return self._hash_encode(text)
    
    def _hash_encode(self, text: str, dim: int = None) -> np.ndarray:
        """Encodage de fallback basé sur hash"""
        target_dim = dim or self.target_dim
        
        hash_obj = hashlib.md5(str(text).encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        
        np.random.seed(hash_int % (2**31))
        features = np.random.normal(0, 0.1, target_dim)
        
        if target_dim >= 4:
            features[0] = len(str(text)) / 100.0
            features[1] = str(text).count('.') / max(len(str(text)), 1)
            features[2] = str(text).count('-') / max(len(str(text)), 1)
            features[3] = float(str(text).isdigit())
        
        return features.astype(np.float32)

class EncoderFactory:
    """Factory pour la gestion des encodeurs spécialisés"""
    
    def __init__(self, target_dim: int = 64):
        self.target_dim = target_dim
        self.encoders = {}
        self.universal_encoder = UniversalTextEncoder(target_dim=target_dim)
        self._initialize_specialized_encoders()
    
    def _initialize_specialized_encoders(self):
        """Initialise tous les encodeurs spécialisés"""
        self.encoders = {
            'IP_ADDRESS': IPAddressEncoder(output_dim=self.target_dim),
            'TIMESTAMP': TimestampEncoder(output_dim=self.target_dim),
            'UUID': UUIDEncoder(output_dim=self.target_dim),
            'SERVICE_5G': Service5GEncoder(output_dim=self.target_dim),
        }
        logger.info(f"Initialized {len(self.encoders)} specialized encoders")
    
    def fit_universal(self, texts: list):
        """Entraîne l'encodeur universel"""
        self.universal_encoder.fit(texts)
    
    def encode_value(self, value: Any, entity_type: str, 
                    confidence: float, metadata: Dict = None) -> np.ndarray:
        """
        Encode une valeur selon son type
        
        Args:
            value: Valeur à encoder
            entity_type: Type d'entité détecté
            confidence: Confiance de la classification
            metadata: Métadonnées additionnelles
            
        Returns:
            np.ndarray: Vecteur encodé
        """
        try:
            # Vérifier si on a un encodeur spécialisé
            if entity_type in self.encoders and confidence >= 0.7:
                encoder = self.encoders[entity_type]
                specialized_vector = encoder.encode(value, metadata)
                
                # Ajouter un flag de confiance
                if len(specialized_vector) < self.target_dim:
                    confidence_feature = np.array([confidence], dtype=np.float32)
                    specialized_vector = np.concatenate([specialized_vector, confidence_feature])
                
                return self._ensure_target_dim(specialized_vector)
            
            else:
                # Utiliser l'encodeur universel
                universal_vector = self.universal_encoder.encode(str(value))
                
                # Ajouter des métadonnées si disponible
                if metadata and len(universal_vector) < self.target_dim:
                    meta_features = self._encode_metadata(metadata)
                    remaining_dim = self.target_dim - len(universal_vector)
                    meta_features = meta_features[:remaining_dim]
                    
                    if len(meta_features) > 0:
                        universal_vector = np.concatenate([universal_vector, meta_features])
                
                return self._ensure_target_dim(universal_vector)
                
        except Exception as e:
            logger.warning(f"Error encoding value '{value}' of type '{entity_type}': {e}")
            return self._fallback_encode(value)
    
    def _encode_metadata(self, metadata: Dict) -> np.ndarray:
        """Encode les métadonnées en features additionnelles"""
        features = []
        
        # Features temporelles si timestamp disponible
        if 'timestamp' in metadata:
            try:
                ts = metadata['timestamp']
                if hasattr(ts, 'hour'):
                    features.extend([
                        ts.hour / 24.0,
                        ts.weekday() / 7.0,
                        ts.day / 31.0
                    ])
            except:
                pass
        
        # Features de classification
        features.extend([
            metadata.get('classification_confidence', 0.0),
            float('network' in str(metadata).lower()),
            float('service' in str(metadata).lower()),
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _ensure_target_dim(self, vector: np.ndarray) -> np.ndarray:
        """S'assure que le vecteur a la dimension cible"""
        current_dim = len(vector)
        
        if current_dim == self.target_dim:
            return vector
        elif current_dim > self.target_dim:
            return vector[:self.target_dim]
        else:
            # Padding avec des zéros
            padding = np.zeros(self.target_dim - current_dim, dtype=np.float32)
            return np.concatenate([vector, padding])
    
    def _fallback_encode(self, value: Any) -> np.ndarray:
        """Encodage de secours en cas d'erreur"""
        try:
            return self.universal_encoder._hash_encode(str(value), self.target_dim)
        except:
            return np.zeros(self.target_dim, dtype=np.float32)
    
    def get_encoder_stats(self) -> Dict[str, Any]:
        """Retourne des statistiques sur les encodeurs"""
        return {
            'specialized_encoders': list(self.encoders.keys()),
            'target_dimension': self.target_dim,
            'universal_encoder_fitted': self.universal_encoder.fitted,
            'total_encoders': len(self.encoders)
        }