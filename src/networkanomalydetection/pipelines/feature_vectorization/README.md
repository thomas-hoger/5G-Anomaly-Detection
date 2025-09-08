# Système de Vectorisation Intelligente pour Graphes 5G

## Vue d'ensemble

Pipeline Kedro pour la vectorisation intelligente de graphes NetworkX contenant des données réseau 5G. Transforme les entités du graphe en vecteurs denses tout en préservant la sémantique métier.

## Utilisation

```bash
kedro run --pipeline=vectorization
```


## Données d'Entrée et de Sortie

### Entrée
- **Fichier trace** : `data/graph/network_graph.pkl`

### Sortie
- **Graphe vectorisé** : `data/05_model_input/vectorized_graph.pkl`

## Modules Principaux

### entity_classifier.py
Classification automatique avec connaissance domaine 5G.

### specialized_encoders.py
Encodeurs optimisés par type d'entité avec fallback TF-IDF.

### hierarchical_handler.py
Traitement spécialisé des structures hiérarchiques 5G.

### intelligent_graph_vectorizer.py
Orchestrateur principal coordonnant le pipeline complet.

### dimension_unifier.py
Unification dimensionnelle avec préservation sémantique.