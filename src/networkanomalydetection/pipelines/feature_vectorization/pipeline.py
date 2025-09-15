from kedro.pipeline import Pipeline, node, pipeline

from .nodes import vectorize_graph_node


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=vectorize_graph_node,
            inputs=["initial_graph", "feature_words", "feature_floats", "params:identifier_list"],
            outputs=["vectorized_graph"],
            name="vectorize_graph"
        )
    ])
