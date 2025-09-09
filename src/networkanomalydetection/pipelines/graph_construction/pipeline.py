from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_graph


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=build_graph,
            inputs="trace_dissected",
            outputs="initial_graph",
            name="trace_to_graph",
        ),
    ])
