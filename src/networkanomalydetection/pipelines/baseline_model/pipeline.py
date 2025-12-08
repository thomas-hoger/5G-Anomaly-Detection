"""
Pipeline d'entraînement GNN pour détection d'anomalies
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import plot_train, test_gnn_model, train_gnn_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_gnn_model,
            inputs=["train_loader", "test_loader", "params:training_params"],
            outputs="gnn_training_results",
            name="train_gnn"
        ),
        node(
            func=plot_train,
            inputs=["gnn_training_results"],
            outputs=None,
            name="plot_train_results"
        ),
        node(
            func=test_gnn_model,
            inputs=["test_loader"],
            outputs="gnn_testing_results",
            name="test_gnn"
        ),
    ])
