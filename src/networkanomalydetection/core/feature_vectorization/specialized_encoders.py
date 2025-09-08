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

    
class HTTPPathEncoder(BaseEncoder):
    """Encodeur spécialisé pour chemins HTTP"""
    
    def __init__(self, output_dim: int = 32):
        super().__init__(output_dim)
        
        # Patterns de chemins API
        self.path_patterns = {
            'API': ['/api/', '/v1/', '/v2/', '/rest/'],
            'AUTH': ['/oauth', '/auth', '/login', '/token'],
            'RESOURCE': ['/users', '/data', '/files', '/items'],
            'ADMIN': ['/admin', '/config', '/settings'],
            'STATIC': ['/css', '/js', '/img', '/assets'],
            '5G_NF': ['/nnrf-', '/nudm-', '/npcf-', '/namf-']
        }
    
    def encode(self, path: str, metadata: Dict = None) -> np.ndarray:
        """Encode un chemin HTTP"""
        try:
            path_str = str(path).lower().strip()
            
            features = [
                # Structure du chemin
                len(path_str) / 100.0,                     # Longueur normalisée
                path_str.count('/') / 10.0,                # Profondeur
                path_str.count('?') / 5.0,                 # Query parameters
                path_str.count('&') / 10.0,                # Multiple params
                path_str.count('=') / 10.0,                # Key-value pairs
                
                # Patterns d'API
                float(any(pattern in path_str for pattern in self.path_patterns['API'])),
                float(any(pattern in path_str for pattern in self.path_patterns['AUTH'])),
                float(any(pattern in path_str for pattern in self.path_patterns['RESOURCE'])),
                float(any(pattern in path_str for pattern in self.path_patterns['ADMIN'])),
                float(any(pattern in path_str for pattern in self.path_patterns['STATIC'])),
                float(any(pattern in path_str for pattern in self.path_patterns['5G_NF'])),
                
                # Types de ressources
                float('/users' in path_str or '/user' in path_str),
                float('/files' in path_str or '/file' in path_str),
                float('/data' in path_str),
                float('/health' in path_str or '/status' in path_str),
                
                # Sécurité et authentification
                float('oauth' in path_str),
                float('token' in path_str),
                float('auth' in path_str),
                float('login' in path_str),
                
                # Features REST
                float(path_str.endswith('/')),             # Trailing slash
                float(path_str.count('/') == 1),           # Root level
                float(any(c.isdigit() for c in path_str)), # Contains IDs
                
                # Hash unique
                (hash(f"path_{path_str}") % 10000) / 10000.0
            ]
            
            # Padding
            while len(features) < self.output_dim:
                features.extend(self._stable_hash_features(path_str, 
                                                         min(5, self.output_dim - len(features))))
                features = features[:self.output_dim]
            
            return np.array(features[:self.output_dim], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error encoding HTTP path {path}: {e}")
            return self._stable_hash_features(str(path), self.output_dim)

class FQDNEncoder(BaseEncoder):
    """Encodeur spécialisé pour noms de domaine"""
    
    def __init__(self, output_dim: int = 32):
        super().__init__(output_dim)
        
        # TLD communs et leurs propriétés
        self.tld_categories = {
            'GENERIC': ['.com', '.org', '.net', '.info'],
            'COUNTRY': ['.fr', '.de', '.uk', '.us', '.jp'],
            'TELECOM': ['.3gppnetwork', '.mnc', '.mcc'],
            'INTERNAL': ['.local', '.internal', '.corp']
        }
        
        # Patterns 5G spécifiques
        self.patterns_5g = ['5gc', 'mnc', 'mcc', '3gppnetwork', 'nrf', 'udm', 'amf']
    
    def encode(self, fqdn: str, metadata: Dict = None) -> np.ndarray:
        """Encode un FQDN"""
        try:
            fqdn_lower = str(fqdn).lower().strip()
            parts = fqdn_lower.split('.')
            
            features = [
                # Structure du domaine
                len(fqdn_lower) / 100.0,                   # Longueur totale
                len(parts) / 10.0,                         # Nombre de sous-domaines
                len(parts[0]) / 20.0 if parts else 0.0,   # Longueur du premier niveau
                
                # Profondeur DNS
                float(len(parts) >= 2),                    # Domain + TLD minimum
                float(len(parts) >= 3),                    # Sous-domaine
                float(len(parts) >= 4),                    # Sous-sous-domaine
                float(len(parts) >= 5),                    # Très profond
                
                # Types de TLD
                float(any(tld in fqdn_lower for tld in self.tld_categories['GENERIC'])),
                float(any(tld in fqdn_lower for tld in self.tld_categories['COUNTRY'])),
                float(any(tld in fqdn_lower for tld in self.tld_categories['TELECOM'])),
                float(any(tld in fqdn_lower for tld in self.tld_categories['INTERNAL'])),
                
                # Patterns 5G
                float(any(pattern in fqdn_lower for pattern in self.patterns_5g)),
                float('5gc' in fqdn_lower),
                float('3gppnetwork' in fqdn_lower),
                float('mnc' in fqdn_lower and 'mcc' in fqdn_lower),
                
                # Services réseau
                float(any(nf in fqdn_lower for nf in ['nrf', 'udm', 'amf', 'smf', 'pcf'])),
                float('api' in fqdn_lower),
                float('service' in fqdn_lower),
                
                # Features de format
                float(fqdn_lower.count('-') > 0),          # Contient des tirets
                float(any(c.isdigit() for c in fqdn_lower)), # Contient des chiffres
                fqdn_lower.count('.') / len(fqdn_lower),   # Densité de points
                
                # Hash unique
                (hash(f"fqdn_{fqdn_lower}") % 10000) / 10000.0
            ]
            
            # Padding
            while len(features) < self.output_dim:
                features.extend(self._stable_hash_features(fqdn_lower, 
                                                         min(5, self.output_dim - len(features))))
                features = features[:self.output_dim]
            
            return np.array(features[:self.output_dim], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error encoding FQDN {fqdn}: {e}")
            return self._stable_hash_features(str(fqdn), self.output_dim)
        
class StatusEncoder(BaseEncoder):
    """Encodeur spécialisé pour états système"""
    
    def __init__(self, output_dim: int = 32):
        super().__init__(output_dim)
        
        # États système avec propriétés
        self.status_properties = {
            'REGISTERED': {'operational': True, 'available': True, 'active': True},
            'ACTIVE': {'operational': True, 'available': True, 'active': True},
            'ENABLED': {'operational': True, 'available': True, 'active': True},
            'SUSPENDED': {'operational': False, 'available': False, 'active': False},
            'DISABLED': {'operational': False, 'available': False, 'active': False},
            'UNDISCOVERABLE': {'operational': True, 'available': False, 'active': True},
            'INACTIVE': {'operational': False, 'available': False, 'active': False},
            'MAINTENANCE': {'operational': False, 'available': False, 'active': False},
        }
    
    def encode(self, status: str, metadata: Dict = None) -> np.ndarray:
        """Encode un état système"""
        try:
            status_upper = str(status).upper().strip()
            props = self.status_properties.get(status_upper, {})
            
            features = [
                # One-hot des états principaux
                float(status_upper == 'REGISTERED'),
                float(status_upper == 'ACTIVE'),
                float(status_upper == 'ENABLED'),
                float(status_upper == 'SUSPENDED'),
                float(status_upper == 'DISABLED'),
                float(status_upper == 'UNDISCOVERABLE'),
                float(status_upper == 'INACTIVE'),
                
                # Propriétés sémantiques
                float(props.get('operational', False)),     # Système opérationnel
                float(props.get('available', False)),       # Service disponible
                float(props.get('active', False)),          # Activement utilisé
                
                # Catégories d'états
                float(status_upper in ['ACTIVE', 'ENABLED', 'REGISTERED']),  # États positifs
                float(status_upper in ['SUSPENDED', 'DISABLED', 'INACTIVE']), # États négatifs
                float(status_upper in ['MAINTENANCE', 'UPDATING']),           # États transitoires
                
                # Features de format
                len(status_upper) / 20.0,                   # Longueur
                float(status_upper in self.status_properties), # État standard
                float('ED' in status_upper),                # Finit par ED (ENABLED, DISABLED)
                float(status_upper.startswith('UN')),       # Préfixe UN (UNDISCOVERABLE)
                
                # Contexte métier
                float(metadata.get('is_nf_status', False)) if metadata else 0.0,
                float(metadata.get('is_service_status', False)) if metadata else 0.0,
                
                # Hash unique
                (hash(f"status_{status_upper}") % 10000) / 10000.0
            ]
            
            # Padding
            while len(features) < self.output_dim:
                features.extend(self._stable_hash_features(status_upper, 
                                                         min(5, self.output_dim - len(features))))
                features = features[:self.output_dim]
            
            return np.array(features[:self.output_dim], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error encoding status {status}: {e}")
            return self._stable_hash_features(str(status), self.output_dim)

class NetworkFlowEncoder(BaseEncoder):
    """Encodeur spécialisé pour relations de flux réseau"""
    
    def encode(self, edge_key: str, metadata: Dict = None) -> np.ndarray:
        """Encode une relation de flux réseau (ip_src, ip_dst, etc.)"""
        try:
            key_lower = str(edge_key).lower().strip()
            
            features = [
                # Types de flux
                float('src' in key_lower or 'source' in key_lower),
                float('dst' in key_lower or 'dest' in key_lower),
                float('ip' in key_lower),
                float('port' in key_lower),
                float('mac' in key_lower),
                
                # Direction du flux
                float(key_lower in ['ip_src', 'src_ip', 'source_ip']),
                float(key_lower in ['ip_dst', 'dst_ip', 'dest_ip']),
                float(key_lower in ['src_port', 'source_port']),
                float(key_lower in ['dst_port', 'dest_port']),
                
                # Protocoles réseau
                float('tcp' in key_lower),
                float('udp' in key_lower),
                float('icmp' in key_lower),
                
                # Features structurelles
                len(key_lower) / 20.0,
                key_lower.count('_') / 5.0,
                float(key_lower.startswith('ip')),
                
                # Hash unique
                (hash(f"netflow_{key_lower}") % 10000) / 10000.0
            ]
            
            # Padding
            while len(features) < self.output_dim:
                features.extend(self._stable_hash_features(key_lower, 
                                                         min(5, self.output_dim - len(features))))
                features = features[:self.output_dim]
            
            return np.array(features[:self.output_dim], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error encoding network flow {edge_key}: {e}")
            return self._stable_hash_features(str(edge_key), self.output_dim)
        
class TemporalEncoder(BaseEncoder):
    """Encodeur spécialisé pour relations temporelles"""
    
    def encode(self, edge_key: str, metadata: Dict = None) -> np.ndarray:
        """Encode une relation temporelle (ts, timestamp, etc.)"""
        try:
            key_lower = str(edge_key).lower().strip()
            
            features = [
                # Types temporels
                float(key_lower == 'ts'),
                float(key_lower == 'timestamp'),
                float('time' in key_lower),
                float('created' in key_lower),
                float('updated' in key_lower),
                float('expires' in key_lower),
                
                # Précision temporelle
                float('ms' in key_lower or 'millis' in key_lower),
                float('ns' in key_lower or 'nanos' in key_lower),
                float('epoch' in key_lower),
                float('utc' in key_lower),
                
                # Contexte événementiel
                float('start' in key_lower),
                float('end' in key_lower),
                float('duration' in key_lower),
                float('interval' in key_lower),
                
                # Features structurelles
                len(key_lower) / 20.0,
                float('_at' in key_lower),  # created_at, updated_at
                
                # Hash unique
                (hash(f"temporal_{key_lower}") % 10000) / 10000.0
            ]
            
            # Padding
            while len(features) < self.output_dim:
                features.extend(self._stable_hash_features(key_lower, 
                                                         min(5, self.output_dim - len(features))))
                features = features[:self.output_dim]
            
            return np.array(features[:self.output_dim], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error encoding temporal relation {edge_key}: {e}")
            return self._stable_hash_features(str(edge_key), self.output_dim)

class PortNumberEncoder(BaseEncoder):
    """Encodeur spécialisé pour ports réseau"""
    
    def __init__(self, output_dim: int = 32):
        super().__init__(output_dim)
        
        # Base de connaissance métier
        self.well_known_ports = {
            21: 'FTP', 22: 'SSH', 23: 'Telnet', 25: 'SMTP', 53: 'DNS',
            80: 'HTTP', 110: 'POP3', 143: 'IMAP', 443: 'HTTPS', 
            993: 'IMAPS', 995: 'POP3S', 587: 'SMTP_TLS'
        }
        
        self.protocol_categories = {
            'WEB': [80, 443, 8080, 8443, 8000, 3000, 3001, 8888],
            'EMAIL': [25, 110, 143, 993, 995, 587, 465],
            'SECURITY': [22, 443, 993, 995, 2222],
            'SYSTEM': [21, 22, 23, 53, 135, 139, 445],
            'DATABASE': [3306, 5432, 1521, 1433, 27017, 6379],
            'DEVELOPMENT': [3000, 3001, 4000, 5000, 8000, 8080, 9000]
        }
    
    def encode(self, port: Union[int, str], metadata: Dict = None) -> np.ndarray:
        """Encode un numéro de port"""
        try:
            port_num = int(port)
            if not (0 <= port_num <= 65535):
                raise ValueError(f"Port {port_num} hors range")
            
            features = [
                # Normalisation position
                port_num / 65535.0,
                math.log(port_num + 1) / math.log(65536),
                
                # Catégories de ports (ranges officiels)
                float(port_num <= 1023),                    # Well-known
                float(1024 <= port_num <= 49151),          # Registered  
                float(port_num >= 49152),                   # Dynamic
                float(port_num in self.well_known_ports),   # Dans notre DB
                
                # Protocoles métier
                float(port_num in self.protocol_categories.get('WEB', [])),
                float(port_num in self.protocol_categories.get('EMAIL', [])),
                float(port_num in self.protocol_categories.get('SECURITY', [])),
                float(port_num in self.protocol_categories.get('SYSTEM', [])),
                float(port_num in self.protocol_categories.get('DATABASE', [])),
                float(port_num in self.protocol_categories.get('DEVELOPMENT', [])),
                
                # Ports spécifiques très communs
                float(port_num == 80),                      # HTTP
                float(port_num == 443),                     # HTTPS
                float(port_num == 22),                      # SSH
                float(port_num == 53),                      # DNS
                
                # Patterns numériques
                float(port_num % 1000 == 80),               # x080 pattern
                float(str(port_num).endswith('443')),       # x443 pattern
                float(port_num % 10 == 0),                  # Ports "ronds"
                float(str(port_num) == str(port_num)[::-1]), # Palindromes
                
                # Features statistiques
                len(str(port_num)) / 5.0,                   # Longueur (1-5 digits)
                str(port_num).count('0') / max(len(str(port_num)), 1),
                str(port_num).count('8') / max(len(str(port_num)), 1),
                
                # Contexte si disponible
                float(metadata.get('is_source_port', False)) if metadata else 0.0,
                float(metadata.get('is_destination_port', False)) if metadata else 0.0,
                
                # Hash unique
                (hash(f"port_{port_num}") % 10000) / 10000.0
            ]
            
            # Padding si nécessaire
            while len(features) < self.output_dim:
                features.extend(self._stable_hash_features(f"port_{port_num}", 
                                                         min(5, self.output_dim - len(features))))
                features = features[:self.output_dim]
            
            return np.array(features[:self.output_dim], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error encoding port {port}: {e}")
            return self._stable_hash_features(str(port), self.output_dim)
        
class HTTPStatusEncoder(BaseEncoder):
    """Encodeur spécialisé pour codes de statut HTTP"""
    
    def __init__(self, output_dim: int = 32):
        super().__init__(output_dim)
        
        # Codes HTTP communs par catégorie
        self.status_categories = {
            'SUCCESS': [200, 201, 202, 204, 206],
            'REDIRECT': [301, 302, 303, 304, 307, 308],
            'CLIENT_ERROR': [400, 401, 403, 404, 405, 409, 410, 413, 414, 429],
            'SERVER_ERROR': [500, 501, 502, 503, 504, 505],
            'INFO': [100, 101, 102],
        }
        
        self.frequent_codes = [200, 301, 302, 400, 401, 403, 404, 500, 502, 503]
    
    def encode(self, status: Union[int, str], metadata: Dict = None) -> np.ndarray:
        """Encode un code de statut HTTP"""
        try:
            status_num = int(status)
            if not (100 <= status_num <= 599):
                raise ValueError(f"HTTP Status {status_num} hors range")
            
            features = [
                # Normalisation dans range HTTP
                (status_num - 100) / 499.0,                # Position dans [100-599]
                status_num / 599.0,                        # Position absolue
                
                # Catégories principales (1XX, 2XX, 3XX, 4XX, 5XX)
                float(100 <= status_num <= 199),           # Informational
                float(200 <= status_num <= 299),           # Success
                float(300 <= status_num <= 399),           # Redirection
                float(400 <= status_num <= 499),           # Client Error
                float(500 <= status_num <= 599),           # Server Error
                
                # Sous-catégories sémantiques
                float(status_num in self.status_categories.get('SUCCESS', [])),
                float(status_num in self.status_categories.get('REDIRECT', [])),
                float(status_num in self.status_categories.get('CLIENT_ERROR', [])),
                float(status_num in self.status_categories.get('SERVER_ERROR', [])),
                
                # Codes très spécifiques
                float(status_num == 200),                  # OK
                float(status_num == 404),                  # Not Found
                float(status_num == 500),                  # Internal Error
                float(status_num == 401),                  # Unauthorized
                float(status_num == 403),                  # Forbidden
                
                # Features de fréquence et criticité
                float(status_num in self.frequent_codes),  # Code fréquent
                float(status_num >= 400),                  # Is error
                float(status_num < 400),                   # Is success/info/redirect
                float(status_num in [503, 504, 502]),      # Service issues
                
                # Patterns numériques
                float(status_num % 100 == 0),              # Round hundreds (200, 300, 400)
                float(str(status_num).endswith('0')),      # Ends with 0
                status_num % 100 / 100.0,                  # Code within category
                
                # Contexte API REST
                float(metadata.get('is_api_response', False)) if metadata else 0.0,
                float(metadata.get('has_body', True)) if metadata else 0.5,
                
                # Hash unique
                (hash(f"http_{status_num}") % 10000) / 10000.0
            ]
            
            # Padding
            while len(features) < self.output_dim:
                features.extend(self._stable_hash_features(f"http_{status_num}", 
                                                         min(5, self.output_dim - len(features))))
                features = features[:self.output_dim]
            
            return np.array(features[:self.output_dim], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error encoding HTTP status {status}: {e}")
            return self._stable_hash_features(str(status), self.output_dim)

class BooleanEncoder(BaseEncoder):
    """Encodeur optimisé pour valeurs booléennes"""
    
    def encode(self, value: Union[bool, str], metadata: Dict = None) -> np.ndarray:
        """Encode une valeur booléenne"""
        try:
            # Normalisation vers booléen
            if isinstance(value, bool):
                bool_val = value
            elif isinstance(value, str):
                bool_val = value.lower() in ['true', '1', 'yes', 'on', 'enabled']
            else:
                bool_val = bool(value)
            
            features = [
                # Encodage principal
                1.0 if bool_val else 0.0,                  # True = 1, False = 0
                0.0 if bool_val else 1.0,                  # False = 1, True = 0 (complement)
                
                # Contexte métier si disponible
                float(metadata.get('is_flag', True)) if metadata else 1.0,
                float(metadata.get('is_status', False)) if metadata else 0.0,
                float(metadata.get('is_config', False)) if metadata else 0.0,
                
                # Features de représentation string
                len(str(value)) / 10.0,                    # Longueur de la représentation
                float(str(value).lower() == 'true'),       # Exactly "true"
                float(str(value).lower() == 'false'),      # Exactly "false"
                float(str(value) in ['1', '0']),           # Numeric representation
                
                # Hash pour différenciation contexte
                (hash(f"bool_{bool_val}_{metadata.get('context', '')}") % 1000) / 1000.0 if metadata else 0.0
            ]
            
            # Padding
            while len(features) < self.output_dim:
                features.append(1.0 if bool_val else 0.0)  # Répéter pattern principal
            
            return np.array(features[:self.output_dim], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error encoding boolean {value}: {e}")
            return self._stable_hash_features(str(value), self.output_dim)       

class DurationEncoder(BaseEncoder):
    """Encodeur spécialisé pour durées en secondes"""
    
    def __init__(self, output_dim: int = 32):
        super().__init__(output_dim)
        
        # Ranges temporels significatifs
        self.time_ranges = {
            'INSTANT': (0, 1),                    # < 1 seconde
            'SECONDS': (1, 60),                   # 1-60 secondes
            'MINUTES': (60, 3600),                # 1-60 minutes  
            'HOURS': (3600, 86400),               # 1-24 heures
            'DAYS': (86400, 604800),              # 1-7 jours
            'WEEKS': (604800, 2592000),           # 1-4 semaines
            'MONTHS': (2592000, 31536000),        # 1-12 mois
            'YEARS': (31536000, float('inf'))     # > 1 an
        }
        
        # Durées typiques
        self.typical_durations = {
            30: 'timeout_short', 60: 'timeout_medium', 300: 'timeout_long',
            3600: 'hour', 86400: 'day', 604800: 'week', 2592000: 'month'
        }
    
    def encode(self, duration: Union[int, float], metadata: Dict = None) -> np.ndarray:
        """Encode une durée en secondes"""
        try:
            dur_seconds = float(duration)
            if dur_seconds < 0:
                raise ValueError(f"Duration {dur_seconds} ne peut pas être négative")
            
            features = [
                # Normalisation logarithmique (durées varient énormément)
                math.log(dur_seconds + 1) / math.log(31536000),  # Log norm sur 1 an
                dur_seconds / 86400.0 if dur_seconds <= 86400 else 1.0,  # Norm sur 1 jour
                
                # Catégories temporelles
                float(self._in_range(dur_seconds, 'INSTANT')),
                float(self._in_range(dur_seconds, 'SECONDS')),
                float(self._in_range(dur_seconds, 'MINUTES')),
                float(self._in_range(dur_seconds, 'HOURS')),
                float(self._in_range(dur_seconds, 'DAYS')),
                float(self._in_range(dur_seconds, 'WEEKS')),
                
                # Durées typiques métier
                float(dur_seconds in self.typical_durations),
                float(dur_seconds == 30),                   # Timeout court
                float(dur_seconds == 300),                  # 5 minutes
                float(dur_seconds == 3600),                 # 1 heure
                float(dur_seconds == 86400),                # 1 jour
                
                # Features contextuelles
                float(dur_seconds % 60 == 0),               # Multiple de minutes
                float(dur_seconds % 3600 == 0),             # Multiple d'heures
                float(dur_seconds % 86400 == 0),            # Multiple de jours
                
                # Contexte métier si disponible
                float(metadata.get('is_timeout', False)) if metadata else 0.0,
                float(metadata.get('is_cache_ttl', False)) if metadata else 0.0,
                float(metadata.get('is_interval', False)) if metadata else 0.0,
                
                # Hash unique
                (hash(f"duration_{int(dur_seconds)}") % 10000) / 10000.0
            ]
            
            # Padding
            while len(features) < self.output_dim:
                features.extend(self._stable_hash_features(f"dur_{int(dur_seconds)}", 
                                                         min(5, self.output_dim - len(features))))
                features = features[:self.output_dim]
            
            return np.array(features[:self.output_dim], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error encoding duration {duration}: {e}")
            return self._stable_hash_features(str(duration), self.output_dim)
    
    def _in_range(self, value: float, range_name: str) -> bool:
        """Vérifie si une valeur est dans un range temporel"""
        min_val, max_val = self.time_ranges[range_name]
        return min_val <= value < max_val

class NetworkFunctionEncoder(BaseEncoder):
    """Encodeur spécialisé pour Network Functions 5G"""
    
    def __init__(self, output_dim: int = 32):
        super().__init__(output_dim)
        
        # Network Functions 5G avec leurs propriétés
        self.nf_properties = {
            'UDM': {'layer': 'CORE', 'criticality': 'HIGH', 'data': True, 'control': True},
            'AMF': {'layer': 'CORE', 'criticality': 'CRITICAL', 'data': False, 'control': True},
            'SMF': {'layer': 'CORE', 'criticality': 'HIGH', 'data': True, 'control': True},
            'PCF': {'layer': 'CORE', 'criticality': 'HIGH', 'data': False, 'control': True},
            'AUSF': {'layer': 'CORE', 'criticality': 'HIGH', 'data': False, 'control': True},
            'NRF': {'layer': 'CORE', 'criticality': 'CRITICAL', 'data': False, 'control': True},
            'UDR': {'layer': 'CORE', 'criticality': 'MEDIUM', 'data': True, 'control': False},
            'NSSF': {'layer': 'CORE', 'criticality': 'MEDIUM', 'data': False, 'control': True},
            'BSF': {'layer': 'CORE', 'criticality': 'LOW', 'data': True, 'control': False},
            'CHF': {'layer': 'CORE', 'criticality': 'MEDIUM', 'data': True, 'control': False},
        }
        
        # Interactions typiques entre NF
        self.nf_interactions = {
            'UDM': ['AMF', 'SMF', 'AUSF', 'UDR'],
            'AMF': ['UDM', 'SMF', 'PCF', 'AUSF', 'NRF'],
            'SMF': ['UDM', 'AMF', 'PCF', 'CHF', 'UDR'],
            'PCF': ['AMF', 'SMF', 'UDR'],
        }
    
    def encode(self, nf_type: str, metadata: Dict = None) -> np.ndarray:
        """Encode un type de Network Function"""
        try:
            nf_upper = str(nf_type).upper().strip()
            props = self.nf_properties.get(nf_upper, {})
            
            features = [
                # One-hot des NF principales
                float(nf_upper == 'UDM'),
                float(nf_upper == 'AMF'),
                float(nf_upper == 'SMF'),
                float(nf_upper == 'PCF'),
                float(nf_upper == 'AUSF'),
                float(nf_upper == 'NRF'),
                float(nf_upper == 'UDR'),
                
                # Propriétés architecturales
                float(props.get('layer') == 'CORE'),
                float(props.get('layer') == 'EDGE'),
                
                # Criticité
                float(props.get('criticality') == 'CRITICAL'),
                float(props.get('criticality') == 'HIGH'),
                float(props.get('criticality') == 'MEDIUM'),
                float(props.get('criticality') == 'LOW'),
                
                # Type de fonction
                float(props.get('data', False)),            # Gère les données
                float(props.get('control', False)),         # Gère le contrôle
                
                # Complexité d'interactions
                len(self.nf_interactions.get(nf_upper, [])) / 10.0,
                
                # Features standards 3GPP
                float(nf_upper in self.nf_properties),      # NF standard
                len(nf_upper) / 5.0,                        # Longueur acronyme
                
                # Contexte 5G si disponible
                float(metadata.get('is_5g_core', True)) if metadata else 1.0,
                float(metadata.get('is_virtualized', True)) if metadata else 1.0,
                
                # Hash unique
                (hash(f"nf_{nf_upper}") % 10000) / 10000.0
            ]
            
            # Padding
            while len(features) < self.output_dim:
                features.extend(self._stable_hash_features(nf_upper, 
                                                         min(5, self.output_dim - len(features))))
                features = features[:self.output_dim]
            
            return np.array(features[:self.output_dim], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error encoding NF type {nf_type}: {e}")
            return self._stable_hash_features(str(nf_type), self.output_dim)

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

class HTTPMethodEncoder(BaseEncoder):
    """Encodeur spécialisé pour méthodes HTTP"""
    
    def __init__(self, output_dim: int = 32):
        super().__init__(output_dim)
        
        # Classification des méthodes HTTP
        self.method_properties = {
            'GET': {'safe': True, 'idempotent': True, 'crud': 'READ', 'frequency': 'HIGH'},
            'POST': {'safe': False, 'idempotent': False, 'crud': 'CREATE', 'frequency': 'HIGH'},
            'PUT': {'safe': False, 'idempotent': True, 'crud': 'UPDATE', 'frequency': 'MEDIUM'},
            'DELETE': {'safe': False, 'idempotent': True, 'crud': 'DELETE', 'frequency': 'MEDIUM'},
            'PATCH': {'safe': False, 'idempotent': False, 'crud': 'UPDATE', 'frequency': 'LOW'},
            'HEAD': {'safe': True, 'idempotent': True, 'crud': 'READ', 'frequency': 'LOW'},
            'OPTIONS': {'safe': True, 'idempotent': True, 'crud': 'META', 'frequency': 'LOW'},
        }
    
    def encode(self, method: str, metadata: Dict = None) -> np.ndarray:
        """Encode une méthode HTTP"""
        try:
            method_upper = str(method).upper().strip()
            props = self.method_properties.get(method_upper, {})
            
            features = [
                # One-hot encoding des méthodes principales
                float(method_upper == 'GET'),
                float(method_upper == 'POST'),
                float(method_upper == 'PUT'),
                float(method_upper == 'DELETE'),
                float(method_upper == 'PATCH'),
                float(method_upper == 'HEAD'),
                float(method_upper == 'OPTIONS'),
                
                # Propriétés sémantiques
                float(props.get('safe', False)),            # Safe methods
                float(props.get('idempotent', False)),      # Idempotent methods
                
                # CRUD mapping
                float(props.get('crud') == 'READ'),
                float(props.get('crud') == 'CREATE'),
                float(props.get('crud') == 'UPDATE'),
                float(props.get('crud') == 'DELETE'),
                float(props.get('crud') == 'META'),
                
                # Fréquence d'usage
                float(props.get('frequency') == 'HIGH'),
                float(props.get('frequency') == 'MEDIUM'),
                float(props.get('frequency') == 'LOW'),
                
                # Features structurelles
                len(method_upper) / 10.0,                   # Longueur méthode
                float(method_upper in self.method_properties),  # Méthode standard
                
                # Contexte API REST si disponible
                float(metadata.get('is_rest_api', True)) if metadata else 1.0,
                float(metadata.get('has_body', method_upper in ['POST', 'PUT', 'PATCH'])) if metadata else float(method_upper in ['POST', 'PUT', 'PATCH']),
                
                # Hash unique
                (hash(f"method_{method_upper}") % 10000) / 10000.0
            ]
            
            # Padding
            while len(features) < self.output_dim:
                features.extend(self._stable_hash_features(method_upper, 
                                                         min(5, self.output_dim - len(features))))
                features = features[:self.output_dim]
            
            return np.array(features[:self.output_dim], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error encoding HTTP method {method}: {e}")
            return self._stable_hash_features(str(method), self.output_dim)

class TokenTypeEncoder(BaseEncoder):
    """Encodeur pour types de tokens d'authentification"""
    
    def __init__(self, output_dim: int = 32):
        super().__init__(output_dim)
        
        self.token_properties = {
            'BEARER': {'strength': 'MEDIUM', 'standard': True, 'stateless': True},
            'BASIC': {'strength': 'LOW', 'standard': True, 'stateless': True},
            'JWT': {'strength': 'HIGH', 'standard': True, 'stateless': True},
            'OAUTH': {'strength': 'HIGH', 'standard': True, 'stateless': False},
            'API_KEY': {'strength': 'MEDIUM', 'standard': False, 'stateless': True},
        }
    
    def encode(self, token_type: str, metadata: Dict = None) -> np.ndarray:
        try:
            token_upper = str(token_type).upper().strip()
            props = self.token_properties.get(token_upper, {})
            
            features = [
                # One-hot des types
                float(token_upper == 'BEARER'),
                float(token_upper == 'BASIC'),
                float(token_upper == 'JWT'),
                float(token_upper == 'OAUTH'),
                float(token_upper == 'API_KEY'),
                
                # Propriétés de sécurité
                float(props.get('strength') == 'HIGH'),
                float(props.get('strength') == 'MEDIUM'),
                float(props.get('strength') == 'LOW'),
                float(props.get('standard', False)),
                float(props.get('stateless', True)),
                
                # Features structurelles
                len(token_upper) / 10.0,
                float('_' in token_upper),
                float(token_upper in self.token_properties),
                
                # Hash unique
                (hash(f"token_{token_upper}") % 10000) / 10000.0
            ]
            
            while len(features) < self.output_dim:
                features.extend(self._stable_hash_features(token_upper, 
                                                         min(5, self.output_dim - len(features))))
                features = features[:self.output_dim]
            
            return np.array(features[:self.output_dim], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error encoding token type {token_type}: {e}")
            return self._stable_hash_features(str(token_type), self.output_dim)

class TelecomCodeEncoder(BaseEncoder):
    """Encodeur pour codes MCC/MNC télécoms"""
    
    def __init__(self, output_dim: int = 32):
        super().__init__(output_dim)
        
        # Codes MCC pays principaux
        self.mcc_countries = {
            '208': 'FR',  # France
            '262': 'DE',  # Allemagne  
            '234': 'GB',  # Royaume-Uni
            '310': 'US',  # États-Unis
            '440': 'JP',  # Japon
        }
        
        # Patterns de validité
        self.valid_mcc_range = range(200, 800)
        self.valid_mnc_range = range(0, 1000)
    
    def encode(self, code: str, metadata: Dict = None) -> np.ndarray:
        try:
            code_str = str(code).strip()
            
            # Déterminer si c'est MCC ou MNC
            is_mcc = len(code_str) == 3 and code_str.isdigit() and int(code_str) in self.valid_mcc_range
            is_mnc = len(code_str) in [2, 3] and code_str.isdigit() and int(code_str) in self.valid_mnc_range
            
            features = [
                # Type de code
                float(is_mcc),
                float(is_mnc),
                float(len(code_str) == 2),  # MNC à 2 chiffres
                float(len(code_str) == 3),  # MCC ou MNC à 3 chiffres
                
                # Validité
                float(code_str.isdigit()),
                float(is_mcc and code_str in self.mcc_countries),  # MCC connu
                
                # Features géographiques (pour MCC)
                float(code_str.startswith('2')),  # Europe (approx)
                float(code_str.startswith('3')),  # Amérique du Nord
                float(code_str.startswith('4')),  # Asie
                float(code_str.startswith('5')),  # Océanie/autres
                
                # Features numériques
                int(code_str) / 999.0 if code_str.isdigit() else 0.0,
                float(code_str.startswith('0')),  # Commence par 0
                
                # Contexte métier
                float(metadata.get('is_mcc', False)) if metadata else float(is_mcc),
                float(metadata.get('is_mnc', False)) if metadata else float(is_mnc),
                
                # Hash unique
                (hash(f"telecom_{code_str}") % 10000) / 10000.0
            ]
            
            while len(features) < self.output_dim:
                features.extend(self._stable_hash_features(code_str, 
                                                         min(5, self.output_dim - len(features))))
                features = features[:self.output_dim]
            
            return np.array(features[:self.output_dim], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error encoding telecom code {code}: {e}")
            return self._stable_hash_features(str(code), self.output_dim)

class HierarchicalFieldEncoder(BaseEncoder):
    """Encodeur pour champs hiérarchiques structurés"""
    
    def encode(self, field: str, metadata: Dict = None) -> np.ndarray:
        try:
            field_str = str(field).strip()
            
            # Analyser la structure hiérarchique
            parts = field_str.split('.')
            brackets = field_str.count('[')
            indices = []
            
            # Extraire les indices entre crochets
            import re
            index_matches = re.findall(r'\[(\d+)\]', field_str)
            indices = [int(match) for match in index_matches]
            
            features = [
                # Structure générale
                len(field_str) / 100.0,                    # Longueur totale
                len(parts) / 10.0,                         # Profondeur
                brackets / 5.0,                            # Nombre d'indices
                field_str.count('.') / 10.0,               # Séparateurs
                
                # Types de structure
                float('[' in field_str and ']' in field_str),  # A des indices
                float(len(parts) > 1),                     # Multi-niveau
                float(any(c.isdigit() for c in field_str)), # Contient nombres
                float('instances' in field_str.lower()),   # Instances d'objets
                
                # Patterns 5G spécifiques
                float('nf' in field_str.lower()),          # Network Function
                float('service' in field_str.lower()),     # Service related
                float('plmn' in field_str.lower()),        # PLMN related
                float('config' in field_str.lower()),      # Configuration
                
                # Complexité
                max(indices) / 100.0 if indices else 0.0, # Index maximum
                np.mean(indices) / 50.0 if indices else 0.0, # Index moyen
                float(len(indices) > 1),                   # Multi-indices
                
                # Hash unique
                (hash(f"hierarchical_{field_str}") % 10000) / 10000.0
            ]
            
            while len(features) < self.output_dim:
                features.extend(self._stable_hash_features(field_str, 
                                                         min(5, self.output_dim - len(features))))
                features = features[:self.output_dim]
            
            return np.array(features[:self.output_dim], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error encoding hierarchical field {field}: {e}")
            return self._stable_hash_features(str(field), self.output_dim)
        
class HTTPProtocolEncoder(BaseEncoder):
    """Encodeur pour relations protocole HTTP"""
    
    def encode(self, edge_key: str, metadata: Dict = None) -> np.ndarray:
        try:
            key_lower = str(edge_key).lower().strip()
            
            features = [
                # Types de relations HTTP
                float(key_lower == 'method'),
                float(key_lower == 'path'),
                float(key_lower == 'status'),
                float(key_lower == 'headers'),
                float(key_lower == 'body'),
                
                # Contexte requête/réponse
                float('request' in key_lower),
                float('response' in key_lower),
                float('header' in key_lower),
                float('param' in key_lower),
                
                # Protocole HTTP spécifique
                float('http' in key_lower),
                float('https' in key_lower),
                float('content' in key_lower),
                float('type' in key_lower),
                
                # Features structurelles
                len(key_lower) / 20.0,
                key_lower.count('_') / 5.0,
                
                # Hash unique
                (hash(f"http_protocol_{key_lower}") % 10000) / 10000.0
            ]
            
            while len(features) < self.output_dim:
                features.extend(self._stable_hash_features(key_lower, 
                                                         min(5, self.output_dim - len(features))))
                features = features[:self.output_dim]
            
            return np.array(features[:self.output_dim], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error encoding HTTP protocol {edge_key}: {e}")
            return self._stable_hash_features(str(edge_key), self.output_dim)

class JWTSecurityEncoder(BaseEncoder):
    """Encodeur pour champs JWT"""
    
    def encode(self, edge_key: str, metadata: Dict = None) -> np.ndarray:
        try:
            key_lower = str(edge_key).lower().strip()
            
            # Extraire le champ JWT (après "jwt.")
            jwt_field = ""
            if key_lower.startswith('jwt.'):
                jwt_field = key_lower[4:]  # Retirer "jwt."
            
            features = [
                # Champs JWT standards
                float(jwt_field == 'sub'),          # Subject
                float(jwt_field == 'aud'),          # Audience
                float(jwt_field == 'exp'),          # Expiration
                float(jwt_field == 'iat'),          # Issued At
                float(jwt_field == 'iss'),          # Issuer
                float(jwt_field == 'nbf'),          # Not Before
                float(jwt_field == 'jti'),          # JWT ID
                
                # Types de claims
                float('scope' in jwt_field),        # OAuth scopes
                float('role' in jwt_field),         # User roles
                float('permission' in jwt_field),   # Permissions
                float('custom' in jwt_field),       # Custom claims
                
                # Sécurité
                float(jwt_field in ['sub', 'aud', 'exp']),  # Critical fields
                float(jwt_field.startswith('priv')),        # Private claims
                float(jwt_field.startswith('pub')),         # Public claims
                
                # Features structurelles
                len(jwt_field) / 20.0,
                float('_' in jwt_field),
                
                # Hash unique
                (hash(f"jwt_{jwt_field}") % 10000) / 10000.0
            ]
            
            while len(features) < self.output_dim:
                features.extend(self._stable_hash_features(key_lower, 
                                                         min(5, self.output_dim - len(features))))
                features = features[:self.output_dim]
            
            return np.array(features[:self.output_dim], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error encoding JWT security {edge_key}: {e}")
            return self._stable_hash_features(str(edge_key), self.output_dim)

class NFRelationEncoder(BaseEncoder):
    """Encodeur pour relations entre Network Functions"""
    
    def encode(self, edge_key: str, metadata: Dict = None) -> np.ndarray:
        try:
            key_lower = str(edge_key).lower().strip()
            
            features = [
                # Types de relations NF
                float('nf' in key_lower),
                float('type' in key_lower),
                float('status' in key_lower),
                float('instance' in key_lower),
                float('service' in key_lower),
                
                # Relations spécifiques 5G
                float('registration' in key_lower),
                float('discovery' in key_lower),
                float('selection' in key_lower),
                float('profile' in key_lower),
                
                # Contexte 5G Core
                float(any(nf in key_lower for nf in ['udm', 'amf', 'smf', 'pcf', 'nrf'])),
                float('3gpp' in key_lower),
                float('sbi' in key_lower),  # Service Based Interface
                
                # Features structurelles
                len(key_lower) / 30.0,
                key_lower.count('_') / 5.0,
                float(key_lower.startswith('nf')),
                
                # Hash unique
                (hash(f"nf_relation_{key_lower}") % 10000) / 10000.0
            ]
            
            while len(features) < self.output_dim:
                features.extend(self._stable_hash_features(key_lower, 
                                                         min(5, self.output_dim - len(features))))
                features = features[:self.output_dim]
            
            return np.array(features[:self.output_dim], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error encoding NF relation {edge_key}: {e}")
            return self._stable_hash_features(str(edge_key), self.output_dim)

class ServiceRelationEncoder(BaseEncoder):
    """Encodeur pour relations de services"""
    
    def encode(self, edge_key: str, metadata: Dict = None) -> np.ndarray:
        try:
            key_lower = str(edge_key).lower().strip()
            
            features = [
                # Types de relations service
                float('service' in key_lower),
                float('endpoint' in key_lower),
                float('interface' in key_lower),
                float('callback' in key_lower),
                float('notification' in key_lower),
                
                # Patterns de service
                float('name' in key_lower),
                float('version' in key_lower),
                float('scheme' in key_lower),
                float('address' in key_lower),
                
                # Contexte API
                float('api' in key_lower),
                float('rest' in key_lower),
                float('resource' in key_lower),
                float('operation' in key_lower),
                
                # Features structurelles
                len(key_lower) / 25.0,
                key_lower.count('_') / 5.0,
                
                # Hash unique
                (hash(f"service_relation_{key_lower}") % 10000) / 10000.0
            ]
            
            while len(features) < self.output_dim:
                features.extend(self._stable_hash_features(key_lower, 
                                                         min(5, self.output_dim - len(features))))
                features = features[:self.output_dim]
            
            return np.array(features[:self.output_dim], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error encoding service relation {edge_key}: {e}")
            return self._stable_hash_features(str(edge_key), self.output_dim)


class StreamMetadataEncoder(BaseEncoder):
    """Encodeur pour métadonnées de flux"""
    
    def encode(self, edge_key: str, metadata: Dict = None) -> np.ndarray:
        try:
            key_lower = str(edge_key).lower().strip()
            
            # Extraire le suffixe après "stream_"
            stream_field = ""
            if key_lower.startswith('stream_'):
                stream_field = key_lower[7:]  # Retirer "stream_"
            
            features = [
                # Types de métadonnées stream
                float(stream_field == 'id'),
                float(stream_field == 'type'),
                float(stream_field == 'source'),
                float(stream_field == 'destination'),
                float(stream_field == 'protocol'),
                
                # Propriétés de flux
                float('data' in stream_field),
                float('control' in stream_field),
                float('media' in stream_field),
                float('real' in stream_field and 'time' in stream_field),
                
                # Qualité de service
                float('qos' in stream_field),
                float('priority' in stream_field),
                float('bandwidth' in stream_field),
                float('latency' in stream_field),
                
                # Features structurelles
                len(stream_field) / 20.0,
                float('_' in stream_field),
                
                # Hash unique
                (hash(f"stream_{stream_field}") % 10000) / 10000.0
            ]
            
            while len(features) < self.output_dim:
                features.extend(self._stable_hash_features(key_lower, 
                                                         min(5, self.output_dim - len(features))))
                features = features[:self.output_dim]
            
            return np.array(features[:self.output_dim], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error encoding stream metadata {edge_key}: {e}")
            return self._stable_hash_features(str(edge_key), self.output_dim)

class ConfigurationEncoder(BaseEncoder):
    """Encodeur pour paramètres de configuration"""
    
    def encode(self, edge_key: str, metadata: Dict = None) -> np.ndarray:
        try:
            key_lower = str(edge_key).lower().strip()
            
            features = [
                # Types de configuration
                float('config' in key_lower),
                float('setting' in key_lower),
                float('parameter' in key_lower),
                float('option' in key_lower),
                float('preference' in key_lower),
                
                # Contexte temporel
                float('validity' in key_lower),
                float('expires' in key_lower),
                float('timeout' in key_lower),
                float('interval' in key_lower),
                
                # Sécurité et auth
                float('oauth' in key_lower),
                float('custom' in key_lower),
                float('secret' in key_lower),
                float('key' in key_lower),
                
                # Scope de configuration
                float('global' in key_lower),
                float('local' in key_lower),
                float('user' in key_lower),
                float('system' in key_lower),
                
                # Features structurelles
                len(key_lower) / 25.0,
                key_lower.count('_') / 5.0,
                
                # Hash unique
                (hash(f"config_{key_lower}") % 10000) / 10000.0
            ]
            
            while len(features) < self.output_dim:
                features.extend(self._stable_hash_features(key_lower, 
                                                         min(5, self.output_dim - len(features))))
                features = features[:self.output_dim]
            
            return np.array(features[:self.output_dim], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error encoding configuration {edge_key}: {e}")
            return self._stable_hash_features(str(edge_key), self.output_dim)

class HierarchicalRelationEncoder(BaseEncoder):
    """Encodeur pour relations hiérarchiques"""
    
    def encode(self, edge_key: str, metadata: Dict = None) -> np.ndarray:
        try:
            key_lower = str(edge_key).lower().strip()
            
            features = [
                # Structure hiérarchique
                float('[' in key_lower and ']' in key_lower),
                float('.' in key_lower),
                key_lower.count('[') / 5.0,
                key_lower.count('.') / 10.0,
                
                # Types d'objets
                float('instance' in key_lower),
                float('array' in key_lower),
                float('list' in key_lower),
                float('object' in key_lower),
                
                # Contexte 5G hiérarchique
                float('nf' in key_lower),
                float('service' in key_lower),
                float('plmn' in key_lower),
                float('slice' in key_lower),
                
                # Profondeur estimée
                len(key_lower.split('.')) / 10.0,
                float(key_lower.count('[') > 1),  # Multi-niveaux
                
                # Features structurelles
                len(key_lower) / 50.0,
                float(any(c.isdigit() for c in key_lower)),
                
                # Hash unique
                (hash(f"hierarchical_rel_{key_lower}") % 10000) / 10000.0
            ]
            
            while len(features) < self.output_dim:
                features.extend(self._stable_hash_features(key_lower, 
                                                         min(5, self.output_dim - len(features))))
                features = features[:self.output_dim]
            
            return np.array(features[:self.output_dim], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error encoding hierarchical relation {edge_key}: {e}")
            return self._stable_hash_features(str(edge_key), self.output_dim)


# ====================================================================
# CLASSIFICATION CONTEXTUELLE POUR RÉSOUDRE LES CONFLITS
# ====================================================================

class ContextAwareEntityClassifier:
    """Classificateur amélioré qui utilise le contexte des arêtes"""
    
    def __init__(self):
        # Hériter de MultiLevelEntityClassifier
        from entity_classifier import MultiLevelEntityClassifier
        self.base_classifier = MultiLevelEntityClassifier()
        self.confidence_threshold = 0.7
    
    def classify_entity(self, value: Any):
        """Classification standard (fallback)"""
        return self.base_classifier.classify_entity(value)
    
    def classify_with_context(self, value: Any, edge_key: str = None, 
                            node_neighbors: List = None):
        """
        Classification intelligente utilisant le contexte
        
        Args:
            value: Valeur à classifier
            edge_key: Clé de l'arête (ex: "port", "status", "method")
            node_neighbors: Nœuds voisins pour contexte additionnel
        
        Returns:
            EntityClassificationResult avec classification contextuelle
        """
        
        # 1. Classification basée sur la clé d'arête (priorité haute)
        if edge_key and isinstance(value, (int, float)):
            context_result = self._classify_by_edge_context(value, edge_key)
            if context_result:
                return context_result
        
        # 2. Classification basée sur les voisins (priorité moyenne)
        if node_neighbors and isinstance(value, (int, float)):
            neighbor_result = self._classify_by_neighbors(value, node_neighbors)
            if neighbor_result:
                return neighbor_result
        
        # 3. Classification standard (fallback)
        return self.classify_entity(value)
    
    def _classify_by_edge_context(self, value: Union[int, float], edge_key: str):
        """Classification basée sur la clé d'arête"""
        from entity_classifier import EntityClassificationResult
        
        # Conversion pour analyse
        if isinstance(value, float):
            int_value = int(value)
        else:
            int_value = value
        
        edge_key_lower = edge_key.lower()
        
        # CONTEXTE PORT (priorité absolue pour ces clés)
        if any(port_key in edge_key_lower for port_key in 
               ['port', 'src_port', 'dst_port', 'source_port', 'destination_port', 'server_port']):
            if 0 <= int_value <= 65535:
                confidence = 0.95 if int_value <= 1023 else 0.85
                return EntityClassificationResult("PORT_NUMBER", confidence, "NETWORK_PORT",
                                                {"context": "edge_key_port", "edge_key": edge_key})
        
        # CONTEXTE HTTP STATUS (priorité absolue pour ces clés)
        if any(status_key in edge_key_lower for status_key in 
               ['status', 'http_status', 'response_code', 'status_code', 'code']):
            if 100 <= int_value <= 599:
                return EntityClassificationResult("HTTP_STATUS", 0.95, f"HTTP_{int_value//100}XX",
                                                {"context": "edge_key_status", "edge_key": edge_key})
        
        # CONTEXTE DURATION (priorité pour ces clés)
        if any(time_key in edge_key_lower for time_key in 
               ['timeout', 'duration', 'ttl', 'expires', 'interval', 'delay']):
            if 1 <= int_value <= 31536000:  # 1 sec à 1 an
                return EntityClassificationResult("DURATION", 0.9, "DURATION_SECONDS",
                                                {"context": "edge_key_time", "edge_key": edge_key})
        
        # CONTEXTE TIMESTAMP
        if any(ts_key in edge_key_lower for ts_key in ['ts', 'timestamp', 'time', 'created_at']):
            if isinstance(value, float) and value > 1600000000:
                return EntityClassificationResult("TIMESTAMP", 0.95, "UNIX_TIMESTAMP",
                                                {"context": "edge_key_timestamp", "edge_key": edge_key})
        
        return None
    
    def _classify_by_neighbors(self, value: Union[int, float], neighbors: List):
        """Classification basée sur les nœuds voisins"""
        from entity_classifier import EntityClassificationResult
        
        neighbor_strings = [str(neighbor).lower() for neighbor in neighbors]
        all_neighbors = ' '.join(neighbor_strings)
        
        int_value = int(value) if isinstance(value, float) else value
        
        # Contexte réseau (IP, protocoles réseau dans les voisins)
        network_indicators = ['ip', 'addr', 'host', 'server', 'client', 'tcp', 'udp']
        has_network_context = any(indicator in all_neighbors for indicator in network_indicators)
        
        # Contexte HTTP (méthodes, paths HTTP dans les voisins)  
        http_indicators = ['get', 'post', 'put', 'delete', 'http', '/api', '/oauth', 'method']
        has_http_context = any(indicator in all_neighbors for indicator in http_indicators)
        
        # Appliquer la logique contextuelle
        if has_network_context and 0 <= int_value <= 65535:
            confidence = 0.8 if int_value <= 1023 else 0.7
            return EntityClassificationResult("PORT_NUMBER", confidence, "NETWORK_PORT",
                                            {"context": "neighbor_network"})
        
        if has_http_context and 100 <= int_value <= 599:
            return EntityClassificationResult("HTTP_STATUS", 0.8, f"HTTP_{int_value//100}XX",
                                            {"context": "neighbor_http"})
        
        return None


class EncoderFactory:
    """Factory pour la gestion des encodeurs spécialisés"""
    
    def __init__(self, target_dim: int = 64):
        self.target_dim = target_dim
        self.encoders = {}
        self.edge_encoders = {} 
        self.universal_encoder = UniversalTextEncoder(target_dim=target_dim)
        self._initialize_specialized_encoders()
    
    def _initialize_specialized_encoders(self):
        """Initialise TOUS les encodeurs spécialisés"""
        self.encoders = {
            # Encodeurs réseau
            'IP_ADDRESS': IPAddressEncoder(output_dim=self.target_dim),
            'PORT_NUMBER': PortNumberEncoder(output_dim=self.target_dim),
            'FQDN': FQDNEncoder(output_dim=self.target_dim),
            
            # Encodeurs temporels
            'TIMESTAMP': TimestampEncoder(output_dim=self.target_dim),
            'DURATION': DurationEncoder(output_dim=self.target_dim),
            
            # Encodeurs HTTP
            'HTTP_STATUS': HTTPStatusEncoder(output_dim=self.target_dim),
            'HTTP_METHOD': HTTPMethodEncoder(output_dim=self.target_dim),
            'HTTP_PATH': HTTPPathEncoder(output_dim=self.target_dim),
            
            # Encodeurs 5G
            'SERVICE_5G': Service5GEncoder(output_dim=self.target_dim),
            'NF_TYPE': NetworkFunctionEncoder(output_dim=self.target_dim),
            'TELECOM_CODE': TelecomCodeEncoder(output_dim=self.target_dim),
            
            # Encodeurs de données
            'UUID': UUIDEncoder(output_dim=self.target_dim),
            'BOOLEAN': BooleanEncoder(output_dim=self.target_dim),
            'STATUS': StatusEncoder(output_dim=self.target_dim),
            
            # Encodeurs de sécurité
            'TOKEN_TYPE': TokenTypeEncoder(output_dim=self.target_dim),
            
            # Encodeurs structurels
            'HIERARCHICAL_FIELD': HierarchicalFieldEncoder(output_dim=self.target_dim),
        }
        
        # Encodeurs d'arêtes spécialisés
        self.edge_encoders = {
            'NETWORK_FLOW': NetworkFlowEncoder(output_dim=self.target_dim),
            'TEMPORAL': TemporalEncoder(output_dim=self.target_dim),
            'HTTP_PROTOCOL': HTTPProtocolEncoder(output_dim=self.target_dim),
            'JWT_SECURITY': JWTSecurityEncoder(output_dim=self.target_dim),
            'NF_RELATION': NFRelationEncoder(output_dim=self.target_dim),
            'SERVICE_RELATION': ServiceRelationEncoder(output_dim=self.target_dim),
            'STREAM_METADATA': StreamMetadataEncoder(output_dim=self.target_dim),
            'CONFIGURATION': ConfigurationEncoder(output_dim=self.target_dim),
            'HIERARCHICAL_RELATION': HierarchicalRelationEncoder(output_dim=self.target_dim),
        }
        
        logger.info(f"Initialized {len(self.encoders)} node encoders and {len(self.edge_encoders)} edge encoders")
        
    def fit_universal(self, texts: list):
        """Entraîne l'encodeur universel"""
        self.universal_encoder.fit(texts)
    
    def encode_value(self, value: Any, entity_type: str, 
                confidence: float, metadata: Dict = None) -> np.ndarray:
        """
        Encode une valeur selon son type - VERSION COMPLÈTE
        
        Args:
            value: Valeur à encoder
            entity_type: Type d'entité détecté
            confidence: Confiance de la classification
            metadata: Métadonnées additionnelles
            
        Returns:
            np.ndarray: Vecteur encodé
        """
        try:
            # Vérifier encodeurs de nœuds
            if entity_type in self.encoders and confidence >= 0.7:
                encoder = self.encoders[entity_type]
                specialized_vector = encoder.encode(value, metadata)
                
                # Ajouter flag de confiance
                if len(specialized_vector) < self.target_dim:
                    confidence_feature = np.array([confidence], dtype=np.float32)
                    specialized_vector = np.concatenate([specialized_vector, confidence_feature])
                
                return self._ensure_target_dim(specialized_vector)
            
            # Vérifier encodeurs d'arêtes (pour EdgeClassifier)
            elif hasattr(self, 'edge_encoders') and entity_type in self.edge_encoders and confidence >= 0.7:
                encoder = self.edge_encoders[entity_type]
                specialized_vector = encoder.encode(value, metadata)
                return self._ensure_target_dim(specialized_vector)
            
            else:
                # Fallback vers encodeur universel
                universal_vector = self.universal_encoder.encode(str(value))
                
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