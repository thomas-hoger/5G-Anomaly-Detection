"""
Classification intelligente multi-niveaux des entités réseau 5G
"""
import re
import logging
from typing import Dict, Any, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class EntityClassificationResult:
    """Résultat de classification d'une entité"""
    
    def __init__(self, entity_type: str, confidence: float, 
                 sub_type: str = None, metadata: Dict = None):
        self.entity_type = entity_type
        self.confidence = confidence
        self.sub_type = sub_type or entity_type
        self.metadata = metadata or {}

class MultiLevelEntityClassifier:
    """Classificateur intelligent multi-niveaux pour entités 5G"""
    
    def __init__(self):
        self.confidence_threshold = 0.7
        self._init_classification_patterns()
    
    def _init_classification_patterns(self):
        """Initialise les patterns de classification"""
        
        # Network Function Types 5G
        self.nf_types = {
            "UDM", "UDR", "PCF", "AMF", "SMF", "AUSF", "NRF", "NSSF", 
            "BSF", "CHF", "SCP", "SEPP", "NEF", "AF", "LMF"
        }
        
        # HTTP Methods
        self.http_methods = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}
        
        # HTTP Status codes range
        self.http_status_range = range(100, 600)
        
        # 5G Service name patterns
        self.service_patterns = [
            r'^n[a-z]{2,4}-[a-z]{2,8}(-[a-z]{2,8})*$',  # nudm-sdm, npcf-am-policy-control
            r'^n[a-z]{3,4}$'  # nrf, nssf
        ]
        
        # Status values
        self.status_values = {
            "REGISTERED", "SUSPENDED", "UNDISCOVERABLE", "ACTIVE", 
            "INACTIVE", "ENABLED", "DISABLED"
        }
    
    def classify_entity(self, value: Any) -> EntityClassificationResult:
        """
        Classification principale d'une entité
        
        Args:
            value: Valeur à classifier
            
        Returns:
            EntityClassificationResult: Résultat de classification
        """
        # Conversion en string pour analyse
        str_value = str(value)
        
        # Classification par type Python d'abord
        if isinstance(value, bool):
            return EntityClassificationResult("BOOLEAN", 1.0, "BOOLEAN_VALUE")
        
        # Classification numérique
        if isinstance(value, (int, float)):
            return self._classify_numeric(value)
        
        # Classification textuelle
        return self._classify_textual(str_value)
    
    def _classify_numeric(self, value: Union[int, float]) -> EntityClassificationResult:
        """Classification des valeurs numériques"""
        
        # Timestamps (Unix timestamp)
        if isinstance(value, float) and value > 1600000000:
            return EntityClassificationResult(
                "TIMESTAMP", 0.95, "UNIX_TIMESTAMP",
                {"timestamp": datetime.fromtimestamp(value)}
            )
        
        # Port numbers
        if isinstance(value, int) and 0 <= value <= 65535:
            confidence = 0.8 if 1 <= value <= 1023 else 0.6  # Well-known ports higher confidence
            return EntityClassificationResult("PORT_NUMBER", confidence, "NETWORK_PORT")
        
        # HTTP Status codes
        if isinstance(value, int) and value in self.http_status_range:
            return EntityClassificationResult("HTTP_STATUS", 0.9, f"HTTP_{value//100}XX")
        
        # Duration values (seconds/milliseconds)
        if isinstance(value, int) and 1 <= value <= 86400:  # 1 second to 1 day
            return EntityClassificationResult("DURATION", 0.7, "DURATION_SECONDS")
        
        # Generic numeric
        return EntityClassificationResult("NUMERIC", 0.3, "GENERIC_NUMBER")
    
    def _classify_textual(self, value: str) -> EntityClassificationResult:
        """Classification des valeurs textuelles"""
        
        # IP Address
        if self._is_ip_address(value):
            sub_type = self._classify_ip_type(value)
            return EntityClassificationResult("IP_ADDRESS", 0.95, sub_type)
        
        # UUID
        if self._is_uuid(value):
            return EntityClassificationResult("UUID", 0.95, "UUID_IDENTIFIER")
        
        # 5G Network Function Type
        if value.upper() in self.nf_types:
            return EntityClassificationResult("NF_TYPE", 0.9, f"NF_{value.upper()}")
        
        # HTTP Method
        if value.upper() in self.http_methods:
            return EntityClassificationResult("HTTP_METHOD", 0.9, f"HTTP_{value.upper()}")
        
        # 5G Service Name
        service_result = self._classify_service_name(value)
        if service_result:
            return service_result
        
        # HTTP Path
        if value.startswith('/'):
            return self._classify_http_path(value)
        
        # FQDN
        if self._is_fqdn(value):
            return EntityClassificationResult("FQDN", 0.8, "DOMAIN_NAME")
        
        # Status values
        if value.upper() in self.status_values:
            return EntityClassificationResult("STATUS", 0.8, f"STATUS_{value.upper()}")
        
        # Token types
        if value in ["Bearer", "Basic", "JWT", "OAuth"]:
            return EntityClassificationResult("TOKEN_TYPE", 0.8, f"TOKEN_{value.upper()}")
        
        # MCC/MNC codes (3 digits)
        if value.isdigit() and len(value) == 3:
            return EntityClassificationResult("TELECOM_CODE", 0.7, "MCC_OR_MNC")
        
        # Hierarchical field (contains brackets and dots)
        if '[' in value and ']' in value and '.' in value:
            return EntityClassificationResult("HIERARCHICAL_FIELD", 0.8, "STRUCTURED_PATH")
        
        # Generic text
        return EntityClassificationResult("TEXT", 0.3, "GENERIC_TEXT")
    
    def _is_ip_address(self, value: str) -> bool:
        """Vérifie si c'est une adresse IP"""
        pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
        if not re.match(pattern, value):
            return False
        
        # Vérifier que chaque octet est valide (0-255)
        octets = value.split('.')
        return all(0 <= int(octet) <= 255 for octet in octets)
    
    def _classify_ip_type(self, ip: str) -> str:
        """Classifie le type d'IP"""
        if ip.startswith('10.'):
            return "PRIVATE_CLASS_A"
        elif ip.startswith('192.168.'):
            return "PRIVATE_CLASS_C"
        elif ip.startswith('172.'):
            return "PRIVATE_CLASS_B"
        elif ip.startswith('127.'):
            return "LOOPBACK"
        else:
            return "PUBLIC_IP"
    
    def _is_uuid(self, value: str) -> bool:
        """Vérifie si c'est un UUID"""
        pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        return bool(re.match(pattern, value, re.IGNORECASE))
    
    def _classify_service_name(self, value: str) -> EntityClassificationResult:
        """Classifie les noms de services 5G"""
        for pattern in self.service_patterns:
            if re.match(pattern, value):
                # Analyser le type de service
                if 'udm' in value:
                    sub_type = "USER_DATA_MANAGEMENT_SERVICE"
                elif 'amf' in value:
                    sub_type = "ACCESS_MANAGEMENT_SERVICE"
                elif 'pcf' in value:
                    sub_type = "POLICY_CONTROL_SERVICE"
                elif 'auth' in value or 'ueau' in value:
                    sub_type = "AUTHENTICATION_SERVICE"
                elif 'disc' in value:
                    sub_type = "DISCOVERY_SERVICE"
                else:
                    sub_type = "GENERIC_5G_SERVICE"
                
                return EntityClassificationResult("SERVICE_5G", 0.85, sub_type)
        
        return None
    
    def _classify_http_path(self, value: str) -> EntityClassificationResult:
        """Classifie les chemins HTTP"""
        if '/oauth2/' in value:
            return EntityClassificationResult("HTTP_PATH", 0.9, "OAUTH_ENDPOINT")
        elif '/nnrf-disc/' in value:
            return EntityClassificationResult("HTTP_PATH", 0.9, "NRF_DISCOVERY_ENDPOINT")
        elif any(nf in value for nf in ['npcf', 'nudm', 'namf']):
            return EntityClassificationResult("HTTP_PATH", 0.8, "NF_SERVICE_ENDPOINT")
        else:
            return EntityClassificationResult("HTTP_PATH", 0.7, "GENERIC_HTTP_PATH")
    
    def _is_fqdn(self, value: str) -> bool:
        """Vérifie si c'est un FQDN"""
        return ('.' in value and 
                not value.replace('.', '').isdigit() and  # Pas une IP
                len(value.split('.')) >= 2 and
                all(part.replace('-', '').isalnum() for part in value.split('.')))

class EdgeClassifier:
    """Classificateur spécialisé pour les arêtes (clés)"""
    
    def __init__(self):
        self.confidence_threshold = 0.7
    
    def classify_edge(self, edge_key: str) -> EntityClassificationResult:
        """
        Classification d'une clé d'arête
        
        Args:
            edge_key: Clé d'arête à classifier
            
        Returns:
            EntityClassificationResult: Classification de l'arête
        """
        key_lower = edge_key.lower()
        
        # Network flow metadata
        if edge_key in ["ip_src", "ip_dst"]:
            return EntityClassificationResult("NETWORK_FLOW", 0.95, f"NETWORK_{edge_key.upper()}")
        
        # Temporal
        if edge_key == "ts":
            return EntityClassificationResult("TEMPORAL", 0.95, "TIMESTAMP_RELATION")
        
        # HTTP Protocol
        if edge_key in ["method", "path", "status"]:
            return EntityClassificationResult("HTTP_PROTOCOL", 0.9, f"HTTP_{edge_key.upper()}")
        
        # JWT Security
        if edge_key.startswith("jwt."):
            jwt_field = edge_key.split(".", 1)[1]
            return EntityClassificationResult("JWT_SECURITY", 0.9, f"JWT_{jwt_field.upper()}")
        
        # 5G NF Relations
        if "nf" in key_lower and ("type" in key_lower or "status" in key_lower):
            return EntityClassificationResult("NF_RELATION", 0.85, "NF_METADATA")
        
        # Service Relations
        if "service" in key_lower:
            return EntityClassificationResult("SERVICE_RELATION", 0.8, "SERVICE_METADATA")
        
        # Stream metadata
        if edge_key.startswith("stream_"):
            return EntityClassificationResult("STREAM_METADATA", 0.8, f"STREAM_{edge_key.split('_', 1)[1].upper()}")
        
        # Hierarchical paths
        if '[' in edge_key and ']' in edge_key:
            return EntityClassificationResult("HIERARCHICAL_RELATION", 0.8, "STRUCTURED_FIELD")
        
        # Configuration
        if any(config_word in key_lower for config_word in ['validity', 'expires', 'custom', 'oauth']):
            return EntityClassificationResult("CONFIGURATION", 0.7, "CONFIG_PARAMETER")
        
        # Generic relation
        return EntityClassificationResult("GENERIC_RELATION", 0.3, "UNKNOWN_RELATION")