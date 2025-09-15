"""
Kedro pipeline for packet dissection
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import trace_dissection


def create_pipeline(**kwargs) -> Pipeline:

    return pipeline([
        node(
            func=trace_dissection,
            inputs={
                "pkt_files": "trace_to_dissect",
                "banned_features": "params:banned_features",
                "label_dataframe_files": "trace_labels"
            },
            outputs="trace_dissected",
            name="dissect_pcap_files"
        )
    ])
