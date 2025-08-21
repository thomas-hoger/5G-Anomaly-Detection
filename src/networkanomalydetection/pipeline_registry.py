"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline
from networkanomalydetection.pipelines.dissection import create_pipeline as create_dissection_pipeline
from networkanomalydetection.pipelines.graph_construction import create_pipeline as create_graph_construction_pipeline
from networkanomalydetection.pipelines.vectorization import create_pipeline as create_vectorization_pipeline 
from networkanomalydetection.pipelines.gnn_conversion import create_pipeline as create_conversion_pipeline
from networkanomalydetection.pipelines.baseline_vectorization import create_pipeline as create_baseline_vectorization_pipeline
from networkanomalydetection.pipelines.gnn_sampling import create_pipeline as create_sampling

def register_pipelines() -> Dict[str, Pipeline]:
    dissection_pipeline = create_dissection_pipeline()
    graph_construction_pipeline = create_graph_construction_pipeline()
    vectorization_pipeline= create_vectorization_pipeline()
    conversion_pipeline= create_conversion_pipeline()
    baseline_vectorization = create_baseline_vectorization_pipeline()
    graph_sampling = create_sampling()
    
    return {
        "dissection": dissection_pipeline,
        "graph_construction": graph_construction_pipeline, 
        "vectorization": vectorization_pipeline,
        "conversion":conversion_pipeline,
        "baseline_vectorization":baseline_vectorization,
        "graph_sampling":graph_sampling,
        "__default__": dissection_pipeline
    }