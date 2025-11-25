"""
Pipeline d'entraînement GNN pour détection d'anomalies
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import plot_train, train_gnn_model


# Dans pipeline.py - version corrigée
def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_gnn_model,
            inputs=["data_loader_1", "data_loader_2", "params:training_params"],
            outputs="gnn_training_results",
            name="train_gnn_autoencoder"
        ),
        node(
            func=plot_train,
            inputs=["gnn_training_results"],
            outputs=None,
            name="plot_train_results"
        )
    ])
