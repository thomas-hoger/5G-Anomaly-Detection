from kedro.pipeline import Pipeline, node, pipeline

from .nodes import graph_to_html


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=graph_to_html,
            inputs="initial_graph",
            outputs="initial_graph_display",
            name="graph_to_html",
        ),
    ])
