"""
Pipeline Kedro pour pré-entraînement GNN
Fichier: src/networkanomalydetection/pipelines/gnn_pretraining/pipeline.py
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    validate_data_node,
    prepare_data_node,
    initialize_model_node,
    train_model_node,
    save_model_node
)

def create_pipeline(**kwargs) -> Pipeline:
    """
    Crée le pipeline de pré-entraînement GNN
    
    Returns:
        Pipeline: Pipeline complet de pré-entraînement
    """
    
    return pipeline([
        
        # Étape 1: Validation des données
        node(
            func=validate_data_node,
            inputs="vectorized_graph",  # Depuis votre pipeline de vectorisation
            outputs="validation_report",
            name="validate_data_node",
            tags=["gnn_pretraining", "validation"]
        ),
        
        # Étape 2: Préparation des données
        node(
            func=prepare_data_node,
            inputs=[
                "vectorized_graph",
                "validation_report",
                "params:gnn_pretraining.data_preparation"
            ],
            outputs=["prepared_data", "preparation_report"],
            name="prepare_data_node",
            tags=["gnn_pretraining", "preparation"]
        ),
        
        # Étape 3: Initialisation du modèle
        node(
            func=initialize_model_node,
            inputs=[
                "preparation_report",
                "params:gnn_pretraining.model"
            ],
            outputs=["initialized_model", "initialization_report"],
            name="initialize_model_node",
            tags=["gnn_pretraining", "model_init"]
        ),
        
        # Étape 4: Entraînement
        node(
            func=train_model_node,
            inputs=[
                "initialized_model",
                "prepared_data",
                "initialization_report",
                "params:gnn_pretraining.training"
            ],
            outputs=["trained_model", "training_report"],
            name="train_model_node",
            tags=["gnn_pretraining", "training", "core"]
        ),
        
        # Étape 5: Sauvegarde
        node(
            func=save_model_node,
            inputs=["trained_model", "training_report"],
            outputs="save_report",
            name="save_model_node",
            tags=["gnn_pretraining", "save"]
        )
        
    ])