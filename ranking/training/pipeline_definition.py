"""Kubeflow Pipelines DAG for retrieval-candidate ranker training."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kfp import dsl

from ranking import config as ranking_config


def _path(root: str, suffix: str) -> str:
    return f"{root}/{suffix}"


def _module_op(
    name: str,
    module: str,
    image: str,
    arguments: list[str],
    working_dir: str,
) -> dsl.ContainerOp:
    op = dsl.ContainerOp(
        name=name,
        image=image,
        command=["python", "-m", module],
        arguments=arguments,
    )
    op.container.working_dir = working_dir
    return op


@dsl.pipeline(
    name=ranking_config.KUBEFLOW_RANKING_PIPELINE_NAME,
    description="Generate retrieval candidates, train LightGBM ranker, evaluate, and push.",
)
def create_ranking_pipeline(
    pipeline_root: str = ranking_config.KUBEFLOW_RANKING_PIPELINE_ROOT,
    raw_data_dir: str = ranking_config.KUBEFLOW_RAW_DATA_DIR,
    image: str = ranking_config.KUBEFLOW_RANKING_IMAGE,
    working_dir: str = "/app",
    candidates_per_user: int = ranking_config.RANKING_CANDIDATES_PER_USER,
    batch_size: int = 4096,
    min_ndcg_at_10: float = ranking_config.MIN_END_TO_END_NDCG_AT_10,
    min_recall_at_100: float = ranking_config.MIN_END_TO_END_RECALL_AT_100,
) -> None:
    """Create the ranking training/evaluation/push DAG."""
    data_dir = _path(pipeline_root, "data")
    artifact_dir = _path(pipeline_root, "artifacts")
    pushed_dir = _path(pipeline_root, "pushed_ranker")

    train_path = _path(data_dir, ranking_config.TRAIN_FILE.name)
    validation_path = _path(data_dir, ranking_config.VALIDATION_FILE.name)
    test_path = _path(data_dir, ranking_config.TEST_FILE.name)
    model_file = _path(artifact_dir, ranking_config.MODEL_FILE.name)
    joblib_model_file = _path(artifact_dir, "lgbm_ranker.joblib")
    features_file = _path(artifact_dir, ranking_config.FEATURES_FILE.name)
    metrics_file = _path(artifact_dir, ranking_config.METRICS_FILE.name)
    end_to_end_metrics_file = _path(artifact_dir, ranking_config.END_TO_END_METRICS_FILE.name)
    scored_candidates_file = _path(artifact_dir, ranking_config.END_TO_END_CANDIDATES_FILE.name)

    prepare_data = _module_op(
        name="prepare-ranking-data",
        module="ranking.training.prepare_ranking_data",
        image=image,
        working_dir=working_dir,
        arguments=[
            "--output-dir",
            data_dir,
            "--raw-data-dir",
            raw_data_dir,
            "--candidates-per-user",
            str(candidates_per_user),
            "--batch-size",
            str(batch_size),
        ],
    )

    train_ranker = _module_op(
        name="train-ranker",
        module="ranking.training.train_ranker",
        image=image,
        working_dir=working_dir,
        arguments=[
            "--train-path",
            train_path,
            "--validation-path",
            validation_path,
            "--model-file",
            model_file,
            "--joblib-model-file",
            joblib_model_file,
            "--features-file",
            features_file,
        ],
    ).after(prepare_data)

    evaluate_ranker = _module_op(
        name="evaluate-ranker",
        module="ranking.evaluate_ranker",
        image=image,
        working_dir=working_dir,
        arguments=[
            "--data-path",
            test_path,
            "--model-file",
            model_file,
            "--output-file",
            metrics_file,
        ],
    ).after(train_ranker)

    evaluate_end_to_end = _module_op(
        name="evaluate-end-to-end",
        module="ranking.evaluate_end_to_end",
        image=image,
        working_dir=working_dir,
        arguments=[
            "--split",
            "test",
            "--candidates-per-user",
            str(candidates_per_user),
            "--model-file",
            model_file,
            "--output-file",
            end_to_end_metrics_file,
            "--scored-candidates-file",
            scored_candidates_file,
            "--batch-size",
            str(batch_size),
            "--raw-data-dir",
            raw_data_dir,
        ],
    ).after(evaluate_ranker)

    _module_op(
        name="push-ranker",
        module="ranking.training.push_ranker",
        image=image,
        working_dir=working_dir,
        arguments=[
            "--model-file",
            model_file,
            "--joblib-model-file",
            joblib_model_file,
            "--features-file",
            features_file,
            "--metrics-file",
            metrics_file,
            "--end-to-end-metrics-file",
            end_to_end_metrics_file,
            "--pushed-dir",
            pushed_dir,
            "--min-ndcg-at-10",
            str(min_ndcg_at_10),
            "--min-recall-at-100",
            str(min_recall_at_100),
        ],
    ).after(evaluate_end_to_end)
