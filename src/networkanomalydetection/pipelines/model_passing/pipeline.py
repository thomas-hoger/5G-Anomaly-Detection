"""
Pipeline d'entraînement GNN pour détection d'anomalies
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    prepare_training_split,
    train_gnn_model,
    generate_training_plots,
    save_trained_model,
)


# Dans pipeline.py - version corrigée
def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # MAIN seulement pour commencer
        node(
            func=prepare_training_split,
            inputs=["training_data_main", "params:data_split.train_ratio", "params:data_split.val_ratio"],
            outputs="train_val_split",
            name="split_train_validation"
        ),
        node(
            func=train_gnn_model,
            inputs=["train_val_split", "params:gnn_model", "params:gnn_training"],
            outputs="gnn_training_results",
            name="train_gnn_autoencoder"
        ),
        node(
            func=generate_training_plots,
            inputs="gnn_training_results",
            outputs=["training_curves_fig", "error_histograms_fig", "training_summary_fig"],
            name="generate_visualizations"
        ),
        node(
            func=save_trained_model,
            inputs="gnn_training_results",
            outputs="model_save_data",
            name="save_model"
        ),
        
    ])