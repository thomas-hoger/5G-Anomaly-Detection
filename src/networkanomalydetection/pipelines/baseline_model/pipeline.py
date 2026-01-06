"""
Pipeline d'entraînement GNN pour détection d'anomalies
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import plot_train, test_gnn_model, train_gnn_model#, plot_pca_latent_space


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_gnn_model,
            inputs=["train_loader", "val_loader", "params:training_params"],
            outputs=["gnn_training_results","last_state_dict"],
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
            inputs=["test_loader","last_state_dict"],
            outputs="gnn_testing_results",
            name="test_gnn"
        ),
        # node(
        #     func=plot_pca_latent_space,
        #     inputs=["train_z_history","gnn_training_results"],
        #     outputs=None,
        #     name="plot_latent_space_train"
        # ),
        # node(
        #     func=plot_pca_latent_space,
        #     inputs=["val_z_history","gnn_training_results"],
        #     outputs=None,
        #     name="plot_latent_space_val"
        # ),
        # node(
        #     func=plot_pca_latent_space,
        #     inputs=["test_z_history","gnn_testing_results"],
        #     outputs=None,
        #     name="plot_latent_space_test"
        # ),
    ])
