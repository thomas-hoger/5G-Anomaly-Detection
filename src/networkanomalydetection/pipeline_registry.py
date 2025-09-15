"""Project pipelines."""

from kedro.pipeline import Pipeline

from networkanomalydetection.pipelines import (
    pre_process,
)


def register_pipelines() -> dict[str, Pipeline]:

    return {
        "pre_process": pre_process.create_pipeline()
    }

