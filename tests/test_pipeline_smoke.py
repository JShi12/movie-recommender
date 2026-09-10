"""Slow smoke tests for the TFX retrieval pipeline wiring.

Run with ``pytest -m pipeline`` (``make test-pipeline``). Requires the
``[pipeline]`` extra (TFX 1.14 / TensorFlow 2.13, Python 3.9-3.10).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.pipeline

tfx = pytest.importorskip("tfx", reason="the [pipeline] extra (TFX) is not installed")


def test_retrieval_pipeline_builds_expected_component_graph(tmp_path):
    from retrieval.training.pipeline_definition import create_pipeline

    pipeline = create_pipeline(
        pipeline_name="smoke",
        pipeline_root=tmp_path / "root",
        metadata_path=tmp_path / "mlmd.sqlite",
        serving_model_dir=tmp_path / "serving",
        train_steps=1,
        eval_steps=1,
    )
    component_types = {type(c).__name__ for c in pipeline.components}
    assert {
        "CsvExampleGen",
        "StatisticsGen",
        "SchemaGen",
        "ExampleValidator",
        "Transform",
        "Trainer",
        "Evaluator",
        "Pusher",
    }.issubset(component_types)


def test_transform_preprocessing_fn_is_importable():
    from retrieval.training import transform_module

    assert callable(transform_module.preprocessing_fn)
