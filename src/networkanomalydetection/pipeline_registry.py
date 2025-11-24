"""Project pipelines."""

from kedro.pipeline import Pipeline

from networkanomalydetection.pipelines import baseline_model, pre_process


def register_pipelines() -> dict[str, Pipeline]:

    return {
        "pre_process": pre_process.create_pipeline(),
        "baseline_model": baseline_model.create_pipeline()
    }

