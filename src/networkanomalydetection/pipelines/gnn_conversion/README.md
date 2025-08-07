# 🔄 Pipeline GNN Conversion

## 📋 Overview

Ce pipeline convertit un graphe NetworkX vectorisé au format PyTorch Geometric, préparant les données pour l'entraînement de réseaux de neurones sur graphes (GNN). Il transforme les données de trafic réseau 5G en format optimisé pour l'apprentissage automatique.

## 🎯 Objectif

**Input** : Graphe NetworkX avec embeddings 64D (nœuds et arêtes)  
**Output** : Données PyTorch Geometric prêtes pour GNN

## 📂 Structure du Pipeline

```
src/networkanomalydetection/
├── core/conversion/                   # Module de conversion
│   ├── __init__.py                   # Exports des classes
│   ├── graph_converter.py            # 🎪 Orchestrateur principal
│   ├── data_extractor.py            # 🔍 Extraction données NetworkX
│   ├── pytorch_builder.py           # 🏗️ Construction PyTorch Geometric
│   └── validator.py                 # 🔍 Validation qualité
└── pipelines/gnn_conversion/         # Interface Kedro
    ├── __init__.py
    ├── nodes.py                      # Fonctions Kedro
    └── pipeline.py                   # Définition pipeline
```

## 🔧 Composants Principaux

### 🎪 GraphConverter - Orchestrateur
**Rôle** : Coordonne tout le processus de conversion
- Gère l'ordre des opérations
- Configuration globale (device, validation)
- Gestion centralisée des erreurs

### 🔍 DataExtractor - Extracteur
**Rôle** : Extrait les données depuis NetworkX
- Embeddings des nœuds (145k × 64D)
- Connexions et embeddings des arêtes (678k × 64D)
- Métadonnées (entity_type, classification_confidence)
- Mapping bidirectionnel node_id ↔ index

### 🏗️ PyTorchBuilder - Constructeur
**Rôle** : Assemble au format PyTorch Geometric
- Conversion numpy → torch tensors
- Format Data object standard
- Optimisations GPU/CPU
- Validation des dimensions

### 🔍 Validator - Validateur
**Rôle** : Garantit la qualité de conversion
- Test dimensions (nœuds/arêtes préservés)
- Test intégrité (pas de NaN, indices valides)
- Test embeddings (correspondance exacte)
- Test structure (topologie préservée)

## 📊 Données d'Entrée

### Format NetworkX attendu :
```python
# Nœuds avec attributs obligatoires :
{
    'label': str,                    # Valeur originale
    'node_type': int,               # 1=central, 2=parameter  
    'packet_id': int,               # Position temporelle
    'embedding': np.array(64),      # Features vectorisées 64D
    'entity_type': str,             # Classification (IP, Service, etc.)
    'classification_confidence': float  # Score confiance
}

# Arêtes avec attributs obligatoires :
{
    'label': str,                   # Type de relation
    'embedding': np.array(64),      # Features vectorisées 64D
    'entity_type': str,            # Classification relation
    'classification_confidence': float  # Score confiance
}
```

## 📤 Données de Sortie

### Format PyTorch Geometric :
```python
{
    'x': torch.Tensor([N, 64]),           # Features nœuds [145341, 64]
    'edge_index': torch.Tensor([2, E]),   # Connexions [2, 678839]
    'edge_attr': torch.Tensor([E, 64]),   # Features arêtes [678839, 64]
    'num_nodes': int,                     # Nombre de nœuds
    'num_edges': int                      # Nombre d'arêtes
}
```

### Métadonnées associées :
- **Mapping nodes** : Correspondance NetworkX ↔ PyTorch
- **Statistiques** : Degrés, densité, dimensions
- **Validation** : Résultats des tests qualité

## 🚀 Utilisation

### 1. Installation des dépendances
```bash
pip install torch numpy networkx
```

### 2. Configuration Catalog (conf/catalog.yml)
```yaml
# Input (existant)
vectorized_graph:
  type: pickle.PickleDataSet
  filepath: data/05_model_input/vectorized_graph.pkl

# Outputs (nouveaux)
gnn_pytorch_data:
  type: pickle.PickleDataSet
  filepath: data/07_model_input/gnn_pytorch_data.pkl

conversion_metadata:
  type: pickle.PickleDataSet
  filepath: data/08_reporting/conversion_metadata.pkl

conversion_validation_report:
  type: json.JSONDataSet
  filepath: data/08_reporting/conversion_validation_report.json

conversion_quality_report:
  type: json.JSONDataSet
  filepath: data/08_reporting/conversion_quality_report.json
```

### 3. Enregistrement Pipeline (pipeline_registry.py)
```python
from networkanomalydetection.pipelines.gnn_conversion import create_pipeline as gnn_conv

def register_pipelines():
    return {
        "gnn_conversion": gnn_conv(),
        "__default__": graph_construction() + gnn_conv()
    }
```

### 4. Exécution
```bash
# Conversion seule
kedro run --pipeline=gnn_conversion

# Pipeline complet
kedro run --pipeline=__default__
```

## 📊 Validation et Qualité

### Tests Automatiques
Le pipeline effectue 4 tests de validation :

1. **Dimension Consistency** : Vérification préservation nombre nœuds/arêtes
2. **Data Integrity** : Contrôle intégrité (NaN, indices valides)
3. **Embeddings Quality** : Correspondance embeddings (seuil 95%)
4. **Graph Structure** : Préservation topologie (seuil 95%)

### Scores de Qualité
- **EXCELLENT** : ≥ 90% tests réussis
- **GOOD** : ≥ 75% tests réussis  
- **ACCEPTABLE** : ≥ 50% tests réussis
- **POOR** : < 50% tests réussis

### Exemple Résultat Validation
```json
{
  "overall_status": "PASSED",
  "tests_passed": 4,
  "tests_failed": 0,
  "quality_score": 1.0,
  "quality_level": "EXCELLENT"
}
```

## 🔍 Monitoring et Debug

### Logs Disponibles
```bash
# Logs détaillés pendant conversion
kedro run --pipeline=gnn_conversion --log-level=DEBUG
```

### Vérification Résultats
```python
# Dans notebook Kedro
context = session.load_context()

# Charger données converties
pytorch_data = context.catalog.load("gnn_pytorch_data")
metadata = context.catalog.load("conversion_metadata")
validation_report = context.catalog.load("conversion_validation_report")

# Vérification rapide
print(f"Conversion: {validation_report['overall_status']}")
print(f"Données: x={pytorch_data['x'].shape}, edges={pytorch_data['edge_index'].shape}")
```

### Diagnostic Problèmes
```python
# Si conversion échoue, consulter :
quality_report = context.catalog.load("conversion_quality_report")
print("Erreurs:", validation_report.get('errors', []))
print("Recommandations:", quality_report.get('recommendations', []))
```

## ⚠️ Prérequis et Limitations

### Prérequis Données
- ✅ Graphe NetworkX avec attributs obligatoires
- ✅ Embeddings de dimension fixe (64D)
- ✅ Pas de valeurs NaN dans les embeddings
- ✅ Types de données cohérents

### Limitations Connues
- **Mémoire** : Chargement complet en RAM (≈500MB pour 145k nœuds)
- **Device** : CPU par défaut, GPU optionnel
- **MultiDiGraph** : Support complet, mais clés d'arêtes simplifiées

## 🚀 Performance

### Temps d'Exécution Typiques
- **Graphe 1k nœuds** : ~2 secondes
- **Graphe 10k nœuds** : ~15 secondes  
- **Graphe 145k nœuds** : ~2-3 minutes

### Optimisations Possibles
- **Batch processing** : Pour graphes > 500k nœuds
- **GPU acceleration** : Conversion directe sur GPU
- **Parallélisation** : Extraction multi-thread

## 📋 Tests Unitaires

```bash
# Lancer tests du module conversion
pytest tests/core/test_conversion.py

# Test conversion basique
pytest tests/core/test_conversion.py::test_conversion_basic

# Test graphe large
pytest tests/core/test_conversion.py::test_large_graph_conversion
```

## 🔄 Intégration Pipeline Global

### Position dans Pipeline Complet
```
JSON Raw Data → Graph Construction → GNN Conversion → GNN Training → Anomaly Detection
     ↑               ↑                    ↑              ↑             ↑
   Input         Graphe NetworkX    PyTorch Data    Trained Model   Predictions
```

### Outputs Utilisés Par
- **Pipeline GNN Training** : `gnn_pytorch_data`
- **Pipeline Model Evaluation** : `conversion_metadata`
- **Reporting & Analytics** : `conversion_quality_report`

## 🛠️ Configuration Avancée

### Options du Convertisseur
```python
# Configuration personnalisée
converter = NetworkXToTorchConverter(
    validate=True,          # Activer validation (recommandé)
    device='cuda',         # GPU si disponible
)
```

### Variables d'Environnement
```bash
# Désactiver validation pour performance
export SKIP_CONVERSION_VALIDATION=true

# Forcer CPU même si GPU disponible
export FORCE_CPU_CONVERSION=true
```

## 📈 Métriques de Suivi

### KPIs Recommandés
- **Taux de réussite** : % conversions sans erreur
- **Score qualité moyen** : Moyenne des scores de validation
- **Temps de traitement** : Durée par 1000 nœuds
- **Taux préservation** : % données préservées exactement

## 🤝 Contribution

### Ajout de Nouveaux Tests
1. Ajouter test dans `core/conversion/validator.py`
2. Mettre à jour seuils dans configuration
3. Documenter nouveau test dans ce README

### Extension pour Nouveaux Formats
1. Implémenter nouveau builder dans `core/conversion/`
2. Ajouter support dans `graph_converter.py`
3. Mettre à jour pipeline nodes

## 📚 Références

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [NetworkX Documentation](https://networkx.org/documentation/)
- [Kedro Pipeline Documentation](https://kedro.readthedocs.io/en/stable/kedro_project_setup/starters.html)

## 🔍 Troubleshooting

### Erreurs Communes

#### 1. "Dimension mismatch in embeddings"
```
Cause: Embeddings de taille différente de 64D
Solution: Vérifier la vectorisation en amont
```

#### 2. "Invalid edge indices"  
```
Cause: Index de nœud supérieur au nombre de nœuds
Solution: Vérifier mapping dans data_extractor
```

#### 3. "NaN values in features"
```
Cause: Embedding contient des valeurs NaN
Solution: Nettoyer données vectorisées en amont
```

### Support
Pour questions spécifiques, consulter :
- Logs détaillés de conversion
- Rapport de validation généré
- Tests unitaires pour exemples d'usage

---

**Version** : 1.0  
**Dernière mise à jour** : août 2025  
**Compatible** : Kedro 0.19+, PyTorch 1.12+, NetworkX 2.6+