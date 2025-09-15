"""
Kedro pipeline for packet dissection
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import trace_labelling


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the dissection pipeline.

    Returns:
        Kedro Pipeline for packet dissection
    """
    return pipeline([
        node(
            func=trace_labelling,
            inputs={
                "pkt_files": "initial_raw_file",
                "evil_ip": "params:evil_ip",
            },
            outputs=["trace_clean", "trace_labels"],
            name="benign"
        )
    ])
