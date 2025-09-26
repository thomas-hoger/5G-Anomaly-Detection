from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    dissection_cleaning,
    dissection_clusterize,
    feature_vectorization,
    graph_building,
    graph_sampling,
    graph_vectorization,
    graph_visualization,
    trace_cleaning_labelling,
    trace_dissection,
    vocabulary_making,
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
            inputs=["trace_to_dissect", "trace_labels"],
            outputs="trace_dissected",
            name="trace_dissection"
        ),
        node(
            func=dissection_cleaning,
            inputs=["trace_dissected", "params:banned_features"],
            outputs="dissected_clean",
            name="dissection_cleaning"
        ),
        node(
            func=vocabulary_making,
            inputs=["dissected_clean", "params:identifier_conversion", "params:nb_cluster"],
            outputs=["feature_words", "feature_floats"],
            name="vocabulary_making"
        ),
        node(
            func=dissection_clusterize,
            inputs=["dissected_clean", "feature_floats", "params:nb_cluster"],
            outputs="dissection_clusteried",
            name="dissection_clusterize"
        ),
        node(
            func=graph_building,
            inputs="dissection_clusteried",
            outputs="initial_graph",
            name="graph_building"
        ),
        node(
            func=graph_visualization,
            inputs="initial_graph",
            outputs="initial_graph_display",
            name="graph_visualization"
        ),
        node(
            func=feature_vectorization,
            inputs=[
                "initial_graph",
                "params:identifier_conversion",
                "feature_words",
            ],
            outputs=["vectorized_features","feature_vectorization_report"],
            name="feature_vectorization"
        ),
        node(
            func=graph_sampling,
            inputs=[
                "vectorized_features",
                "params:window_size",
                "params:window_shift",
            ],
            outputs=["subgraphs","sampling_report"],
            name="graph_sampling"
        ),
        node(
            func=graph_vectorization,
            inputs=[
                "subgraphs",
                "params:batch_size",
                "params:split_ratio",
            ],
            outputs=["data_loader_1", "data_loader_2"],
            name="graph_vectorization"
        )
    ])


