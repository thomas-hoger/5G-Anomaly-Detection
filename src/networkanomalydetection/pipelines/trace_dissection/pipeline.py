"""
Kedro pipeline for packet dissection
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import trace_dissection


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the dissection pipeline.

    Returns:
        Kedro Pipeline for packet dissection
    """
    return pipeline([
        node(
            func=trace_dissection,
            inputs={
                "pkts": "trace_to_dissect",
                "banned_features": "params:banned_features",
                "label_dataframe": "trace_labels"
            },
            outputs="trace_dissected",
            name="dissect_pcap_files"
        )
    ])
