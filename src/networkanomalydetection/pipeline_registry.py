"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline
from networkanomalydetection.pipelines.dissection import create_pipeline as create_dissection_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    dissection_pipeline = create_dissection_pipeline()
    
    return {
        "dissection": dissection_pipeline,
        "__default__": dissection_pipeline
    }