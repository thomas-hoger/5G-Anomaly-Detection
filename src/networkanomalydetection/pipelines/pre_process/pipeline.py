from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    feature_splitting,
    feature_vectorization,
    graph_building,
    graph_sampling,
    graph_vectorization,
    graph_visualization,
    trace_cleaning_labelling,
    trace_dissection,
)


def create_pipeline(**kwargs) -> Pipeline:

    return pipeline([

        node(
            func=trace_cleaning_labelling,
            inputs={
                "pkt_files": "initial_raw_file",
                "evil_ip": "params:evil_ip",
            },
            outputs=["trace_clean", "trace_labels"],
            name="trace_cleaning_labelling"
        ),
        node(
            func=trace_dissection,
            inputs={
                "pkt_files": "trace_to_dissect",
                "banned_features": "params:banned_features",
                "label_dataframe_files": "trace_labels"
            },
            outputs="trace_dissected",
            name="trace_dissection"
        ),
        node(
            func=graph_building,
            inputs={"dissected_files": "trace_dissected"},
            outputs="initial_graph",
            name="graph_building"
        ),
        node(
            func=graph_visualization,
            inputs={"graph_files": "initial_graph"},
            outputs="initial_graph_display",
            name="graph_visualization"
        ),
        node(
            func=feature_splitting,
            inputs={
                "graph_files": "initial_graph",
                "identifier_conversion": "params:identifier_conversion",
            },
            outputs=["feature_words", "feature_floats"],
            name="feature_splitting"
        ),
        node(
            func=feature_vectorization,
            inputs={
                "graph_files": "initial_graph",
                "feature_words": "feature_words",
                "feature_floats": "feature_floats",
                "identifier_conversion": "params:identifier_conversion",
            },
            outputs=["vectorized_features","feature_vectorization_report"],
            name="feature_vectorization"
        ),
        node(
            func=graph_sampling,
            inputs={
                "graph_files": "vectorized_features",
                "window_size": "params:window_size",
                "window_shift": "params:window_shift",
            },
            outputs=["subgraphs","sampling_report"],
            name="graph_sampling"
        ),
        node(
            func=graph_vectorization,
            inputs={
                "graph_files": "subgraphs",
                "batch_size": "params:batch_size",
                "split_ratio": "params:split_ratio",
            },
            outputs=["data_loader_1", "data_loader_2"],
            name="graph_vectorization"
        )
    ])


