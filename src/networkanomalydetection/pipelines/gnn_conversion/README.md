#  Pipeline Kedro — Conversion Graphe NetworkX → PyTorch Geometric

##  Description
Ce pipeline, nommé **`gnn_conversion`**, a pour but de transformer un **graphe vectorisé** (au format NetworkX avec ses features) en un objet **`torch_geometric.data.Data`** prêt à être utilisé directement dans un modèle de **Graph Neural Network (GNN)** sous PyTorch Geometric.

Il intègre également une étape de **validation qualité** afin de vérifier l’intégrité et la cohérence des données converties.

---

##  Fonctionnalités principales
1. **Conversion des graphes vectorisés** (features des nœuds/arêtes inclus) en objets PyTorch Geometric (`Data`).
2. **Production de métadonnées** de conversion (dimensions, structure, etc.).
3. **Vérification de la qualité** de la conversion (cohérence des dimensions, correspondance entre features et structure).
4. **Support d’un graphe vectorisé de 2 manieres**  pour comparaison.

---

##  Entrées
- `vectorized_graph` : Graphe NetworkX vectorisé (nœuds, arêtes, features).
- `baseline_vectorized_graph` : Graphe baseline vectorisé (GMM+TF-IDF, pour comparaison).

---

##  Sorties
- `gnn_pytorch_data` : Objet PyTorch Geometric prêt pour un modèle.
- `conversion_metadata` : Informations sur la conversion (nb de nœuds, nb d’arêtes, dimensions des features…).
- `conversion_quality_report` : Rapport qualité.
- Équivalents pour la baseline :
  - `baseline_gnn_pytorch_data`
  - `baseline_conversion_metadata`
  - `baseline_conversion_quality_report`

---



# Exécution via Kedro CLI
# kedro run --pipeline=gnn_conversion
