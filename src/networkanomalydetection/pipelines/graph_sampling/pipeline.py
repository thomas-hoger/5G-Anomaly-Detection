"""
Pipeline GNN Sampling
"""
from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    generate_subgraphs_main,
    generate_subgraphs_baseline,
    prepare_training_data_main,
    prepare_training_data_baseline
)

def create_pipeline(**kwargs) -> Pipeline:
    """Pipeline génération subgraphs pour entraînement GNN"""
    
    return pipeline([
        
        # Génération subgraphs principal
        node(
            func=generate_subgraphs_main,
            inputs="vectorized_graph",
            outputs="subgraphs_main_raw",
            name="generate_main_subgraphs"
        ),
        
        # Préparation données principal
        node(
            func=prepare_training_data_main,
            inputs="subgraphs_main_raw",
            outputs="training_data_main",
            name="prepare_main_data"
        ),
        
        # Génération subgraphs baseline
        node(
            func=generate_subgraphs_baseline,
            inputs="baseline_vectorized_graph",
            outputs="subgraphs_baseline_raw",
            name="generate_baseline_subgraphs"
        ),
        
        # Préparation données baseline
        node(
            func=prepare_training_data_baseline,
            inputs="subgraphs_baseline_raw",
            outputs="training_data_baseline",
            name="prepare_baseline_data"
        )
    ])