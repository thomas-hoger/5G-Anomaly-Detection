from kedro.pipeline import Pipeline, node, pipeline

from .nodes import graph_to_tensor


def create_pipeline(**kwargs) -> Pipeline:

    return pipeline([
        node(
            func=graph_to_tensor,
            inputs="vectorized_graph",
            outputs=["train_loader","val_loader"],
            name="convert_to_pytorch_node"
        )
    ])
