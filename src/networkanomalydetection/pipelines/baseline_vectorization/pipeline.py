"""
This is a boilerplate pipeline 'baseline_vectorization'
generated using Kedro 1.0.0
"""
"""
Pipeline de vectorisation baseline : GMM + TF-IDF
"""
from kedro.pipeline import Pipeline, pipeline, node
from .nodes import baseline_vectorize_graph_node,baseline_generate_report_node

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=baseline_vectorize_graph_node,
            inputs="complete_graph_pickle",
            outputs="baseline_vectorized_graph", 
            name="baseline_vectorize_graph_node",
        ),
        node(
            func=baseline_generate_report_node,
            inputs="complete_graph_pickle",
            outputs="baseline_vectorization_report",  
            name="baseline_generate_report_node",
        ),
    
    ])