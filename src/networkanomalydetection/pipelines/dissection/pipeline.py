"""
Kedro pipeline for packet dissection
"""
from kedro.pipeline import Pipeline, pipeline, node
from .nodes import process_pcap_files


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the dissection pipeline.
    
    Returns:
        Kedro Pipeline for packet dissection
    """
    return pipeline([
        node(
            func=process_pcap_files,
            inputs={
                "input_trace_dir": "params:input_trace_dir",
                "banned_features": "params:banned_features",
                "buffer_size": "params:buffer_size"
            },
            outputs="dissected_data",
            name="dissect_pcap_files"
        )
    ])