"""
This is a boilerplate pipeline 'graph_construction'
generated using Kedro 1.0.0rc1
"""
from kedro.pipeline import Pipeline, pipeline, node
from .nodes import process_trace_file

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=process_trace_file,
            inputs="input_trace_file",
            outputs=["graph_html_output", "complete_graph_pickle"],  
            name="process_trace_to_graph_node",
        ),
    ])