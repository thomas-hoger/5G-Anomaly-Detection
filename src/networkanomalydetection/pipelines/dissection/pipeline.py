"""
This is a boilerplate pipeline 'dissection'
generated using Kedro 0.19.14
"""
"""
This is a boilerplate pipeline 'dissection'
generated using Kedro 0.19.14
"""

from kedro.pipeline import node, Pipeline, pipeline

from .nodes import dissect_pcap_files, generate_dissection_report

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=dissect_pcap_files,
                inputs=["parameters"],
                outputs="dissected_packets",
                name="dissect_pcap_files_node",
                tags=["dissection"]
            ),
            node(
                func=generate_dissection_report,
                inputs=["dissected_packets", "parameters"],
                outputs="dissection_report", 
                name="generate_report_node",
                tags=["reporting"]
            )
        ]
    )