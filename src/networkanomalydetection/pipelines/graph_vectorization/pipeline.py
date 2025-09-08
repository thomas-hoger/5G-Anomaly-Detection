"""
This is a boilerplate pipeline 'gnn_conversion'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import convert_networkx_to_pytorch, validate_conversion_quality

def create_pipeline(**kwargs) -> Pipeline:
    """
    Crée le pipeline de conversion GNN
    
    Returns:
        Pipeline Kedro pour conversion NetworkX → PyTorch Geometric
    """
    return pipeline([
        node(
            func=convert_networkx_to_pytorch,
            inputs="vectorized_graph",
            outputs=["gnn_pytorch_data", "conversion_metadata", "conversion_validation_report"],
            name="convert_to_pytorch_node"
        ),
        node(
            func=validate_conversion_quality,
            inputs="conversion_validation_report", 
            outputs="conversion_quality_report",
            name="validate_quality_node"
        ),
        node(
            func=convert_networkx_to_pytorch,
            inputs="baseline_vectorized_graph",
            outputs=["baseline_gnn_pytorch_data", "baseline_conversion_metadata", "baseline_conversion_validation_report"],
            name="convert_baseline_to_pytorch_node"
        ),
        node(
            func=validate_conversion_quality,
            inputs="baseline_conversion_validation_report", 
            outputs="baseline_conversion_quality_report",
            name="validate_baseline_quality_node"
        )
        

        
    ])