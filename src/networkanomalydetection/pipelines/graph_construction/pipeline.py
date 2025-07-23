"""
This is a boilerplate pipeline 'graph_construction'
generated using Kedro 1.0.0rc1
"""
from kedro.pipeline import node, Pipeline, pipeline
from .nodes import build_complete_topology_graph, extract_comprehensive_sequences, generate_comprehensive_graph_report

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build_complete_topology_graph,
                inputs=["dissected_packets", "parameters"],  
                outputs="complete_topology_graph",
                name="build_complete_topology_graph_node",
                tags=["graph_construction", "topology", "complete"]
            ),
            node(
                func=extract_comprehensive_sequences,
                inputs=["complete_topology_graph", "parameters"],
                outputs="complete_graph_sequences",
                name="extract_comprehensive_sequences_node",
                tags=["graph_construction", "sequences", "analysis"]
            ),
            node(
                func=generate_comprehensive_graph_report,
                inputs=["complete_topology_graph", "complete_graph_sequences", "parameters"],
                outputs="complete_graph_report",
                name="generate_comprehensive_graph_report_node",
                tags=["reporting", "graph", "analysis"]
            )
        ]
    )