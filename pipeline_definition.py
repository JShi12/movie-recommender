"""TFX pipeline definition for the MovieLens recommender."""

from __future__ import annotations

from pathlib import Path

import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    Evaluator,
    ExampleValidator,
    Pusher,
    SchemaGen,
    StatisticsGen,
    Trainer,
    Transform,
)
from tfx.dsl.components.common import resolver
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import metadata, pipeline
from tfx.proto import example_gen_pb2, pusher_pb2, trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

import config


def create_eval_config() -> tfma.EvalConfig:
    """Create the TFMA evaluation config used by the Evaluator component."""
    return tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key="label")],
        slicing_specs=[
            tfma.SlicingSpec(),
        ],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(
                        class_name="AUC",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={"value": 0.6}
                            ),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={"value": 0.0},
                            ),
                        ),
                    ),
                    tfma.MetricConfig(class_name="Precision"),
                    tfma.MetricConfig(class_name="Recall"),
                    tfma.MetricConfig(class_name="BinaryAccuracy"),
                    tfma.MetricConfig(class_name="ConfusionMatrixPlot"),
                    tfma.MetricConfig(class_name="CalibrationPlot"),
                    tfma.MetricConfig(
                        class_name="Precision",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={"value": 0.6}
                            )
                        ),
                    ),
                ]
            )
        ],
    )


def create_pipeline(
    pipeline_name: str = config.PIPELINE_NAME,
    pipeline_root: str | Path = config.PIPELINE_ROOT,
    data_root: str | Path = config.DATA_ROOT,
    transform_module_path: str | Path = config.TRANSFORM_MODULE_FILE,
    trainer_module_path: str | Path = config.TRAINER_MODULE_FILE,
    serving_model_dir: str | Path = config.SERVING_MODEL_DIR,
    metadata_path: str | Path = config.METADATA_PATH,
    train_steps: int = config.TRAIN_STEPS,
    eval_steps: int = config.EVAL_STEPS,
) -> pipeline.Pipeline:
    """Create the full TFX pipeline."""
    trainer_custom_config = dict(config.TRAINER_HYPERPARAMETERS)

    example_gen = CsvExampleGen(
        input_base=str(data_root),
        input_config=example_gen_pb2.Input(
            splits=[
                example_gen_pb2.Input.Split(name="train", pattern="train.csv"),
                example_gen_pb2.Input.Split(name="eval", pattern="val.csv"),
                example_gen_pb2.Input.Split(name="test", pattern="test.csv"),
            ]
        ),
    )

    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])
    schema_gen = SchemaGen(statistics=statistics_gen.outputs["statistics"])
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=str(Path(transform_module_path).resolve()),
    )

    trainer = Trainer(
        module_file=str(Path(trainer_module_path).resolve()),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(num_steps=train_steps),
        eval_args=trainer_pb2.EvalArgs(num_steps=eval_steps),
        custom_config=trainer_custom_config,
    )

    model_resolver = resolver.Resolver(
        strategy_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    ).with_id("latest_blessed_model_resolver")

    evaluator = Evaluator(
        examples=transform.outputs["transformed_examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=create_eval_config(),
        example_splits=["eval"],
    )

    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=str(serving_model_dir)
            )
        ),
    )

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=str(pipeline_root),
        components=[
            example_gen,
            statistics_gen,
            schema_gen,
            example_validator,
            transform,
            trainer,
            model_resolver,
            evaluator,
            pusher,
        ],
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            str(metadata_path)
        ),
        enable_cache=True,
    )
