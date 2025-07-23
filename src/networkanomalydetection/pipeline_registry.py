"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline
from networkanomalydetection.pipelines.dissection import create_pipeline as create_dissection_pipeline
from networkanomalydetection.pipelines.graph_construction import create_pipeline as create_graph_construction_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    dissection_pipeline = create_dissection_pipeline()
    graph_construction_pipeline = create_graph_construction_pipeline()
    
    return {
        "dissection": dissection_pipeline,
        "graph_construction": graph_construction_pipeline, 
        "__default__": dissection_pipeline
    }