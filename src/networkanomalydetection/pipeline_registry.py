"""Project pipelines."""

from kedro.pipeline import Pipeline

from networkanomalydetection.pipelines import (
    feature_split_type,
    feature_vectorization,
    graph_construction,
    graph_sampling,
    graph_visualization,
    # graph_vectorization,
    # model_passing,
    trace_cleaning_and_labelling,
    trace_dissection,
)


def register_pipelines() -> dict[str, Pipeline]:

    return {
        "trace_cleaning_and_labelling" : trace_cleaning_and_labelling.create_pipeline(),
        "trace_dissection": trace_dissection.create_pipeline(),
        "graph_construction": graph_construction.create_pipeline(),
        "graph_visualization": graph_visualization.create_pipeline(),
        "feature_split_type": feature_split_type.create_pipeline(),
        "feature_vectorization": feature_vectorization.create_pipeline(),
        "graph_sampling": graph_sampling.create_pipeline(),
        # "graph_vectorization":graph_vectorization.create_pipeline(),
        # "model_passing":model_passing.create_pipeline()
    }

