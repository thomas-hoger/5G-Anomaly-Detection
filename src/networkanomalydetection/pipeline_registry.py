"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline
from networkanomalydetection.pipelines.dissection import create_pipeline as create_dissection_pipeline
from networkanomalydetection.pipelines.graph_construction import create_pipeline as create_graph_construction_pipeline
from networkanomalydetection.pipelines.vectorization import create_pipeline as create_vectorization_pipeline 
from networkanomalydetection.pipelines.gnn_pretraining import create_pipeline as create_gnn_pretraining_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    dissection_pipeline = create_dissection_pipeline()
    graph_construction_pipeline = create_graph_construction_pipeline()
    vectorization_pipeline= create_vectorization_pipeline()
    gnn_pretraining_pipeline= create_gnn_pretraining_pipeline()
    
    return {
        "dissection": dissection_pipeline,
        "graph_construction": graph_construction_pipeline, 
        "vectorization": vectorization_pipeline,
        "gnn_pretraining":gnn_pretraining_pipeline,
        "__default__": dissection_pipeline
    }