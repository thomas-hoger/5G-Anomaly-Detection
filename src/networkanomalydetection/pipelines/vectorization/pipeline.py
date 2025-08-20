"""
This is a boilerplate pipeline 'vectorization' generated using Kedro 1.0.0rc1
"""
from kedro.pipeline import Pipeline, pipeline, node
from .nodes import vectorize_graph_node, create_vectorization_summary_node

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=vectorize_graph_node,
            inputs=["complete_graph_pickle"],
            outputs=["vectorized_graph", "vectorization_report"],
            name="vectorize_graph_node",
            tags=["vectorization", "machine_learning", "core"]
        ),
        node(
            func=create_vectorization_summary_node,
            inputs="vectorization_report",
            outputs="vectorization_summary",
            name="create_vectorization_summary_node",
            tags=["vectorization", "reporting", "monitoring"]
        ),
    ])