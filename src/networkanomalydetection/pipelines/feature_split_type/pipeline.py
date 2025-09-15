from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_by_type


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_by_type,
            inputs=["initial_graph", "params:identifier_list"],
            outputs=["feature_words", "feature_floats"],
            name="split_feature_by_type"
        )
    ])
