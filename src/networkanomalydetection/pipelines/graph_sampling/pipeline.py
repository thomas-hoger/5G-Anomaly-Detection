from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generate_subgraphs


def create_pipeline(**kwargs) -> Pipeline:
    """Pipeline génération subgraphs pour entraînement GNN"""

    return pipeline([

        # Génération subgraphs principal
        node(
            func=generate_subgraphs,
            inputs=["vectorized_graph", "params:subgraph_history_length"],
            outputs="subgraphs",
            name="generate_subgraphs"
        )
    ])
