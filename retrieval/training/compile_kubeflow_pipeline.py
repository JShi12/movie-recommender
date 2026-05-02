"""Compile the TFX pipeline to a Kubeflow Pipelines YAML package."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import kfp
from kfp.dsl import _container_op as kfp_container_op
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.orchestration.kubeflow.kubeflow_dag_runner import KubeflowDagRunnerConfig

from retrieval import config
from retrieval.training.pipeline_definition import create_pipeline


def ensure_kfp_compile_flag() -> None:
    """Ensure COMPILING_FOR_V2 exists for both kfp and container-op module refs."""
    if not hasattr(kfp, "COMPILING_FOR_V2"):
        kfp.COMPILING_FOR_V2 = False
    if not hasattr(kfp_container_op.kfp, "COMPILING_FOR_V2"):
        kfp_container_op.kfp.COMPILING_FOR_V2 = False


def looks_like_placeholder_gcs(path: str) -> bool:
    return path.startswith("gs://") and "your-bucket" in path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-file", type=Path, default=config.KUBEFLOW_PIPELINE_FILE)
    parser.add_argument("--pipeline-name", default=config.KUBEFLOW_PIPELINE_NAME)
    parser.add_argument("--pipeline-root", default="gs://your-bucket/tfx_pipeline")
    parser.add_argument("--data-root", default="gs://your-bucket/data/retrieval")
    parser.add_argument("--serving-model-dir", default=str(config.SERVING_MODEL_DIR))
    parser.add_argument("--metadata-path", default=str(config.METADATA_PATH))
    parser.add_argument("--tfx-image", default=config.KUBEFLOW_TFX_IMAGE)
    parser.add_argument("--transform-module", type=Path, default=config.TRANSFORM_MODULE_FILE)
    parser.add_argument("--trainer-module", type=Path, default=config.TRAINER_MODULE_FILE)
    parser.add_argument("--train-steps", type=int, default=config.TRAIN_STEPS)
    parser.add_argument("--eval-steps", type=int, default=config.EVAL_STEPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_kfp_compile_flag()

    pipeline_root = args.pipeline_root
    data_root = args.data_root
    if looks_like_placeholder_gcs(pipeline_root) or looks_like_placeholder_gcs(data_root):
        pipeline_root = str(config.PIPELINE_ROOT)
        data_root = str(config.RETRIEVAL_DATA_DIR)
        print("Placeholder GCS paths detected; compiling with local paths.")

    tfx_pipeline = create_pipeline(
        pipeline_name=args.pipeline_name,
        pipeline_root=pipeline_root,
        data_root=data_root,
        transform_module_path=os.path.abspath(args.transform_module),
        trainer_module_path=os.path.abspath(args.trainer_module),
        serving_model_dir=args.serving_model_dir,
        metadata_path=args.metadata_path,
        train_steps=args.train_steps,
        eval_steps=args.eval_steps,
    )

    runner_config = KubeflowDagRunnerConfig(
        kubeflow_metadata_config=kubeflow_dag_runner.get_default_kubeflow_metadata_config(),
        tfx_image=args.tfx_image,
    )
    runner = kubeflow_dag_runner.KubeflowDagRunner(
        config=runner_config,
        output_filename=str(args.output_file),
    )
    runner.run(tfx_pipeline)
    print(f"Kubeflow pipeline compiled: {args.output_file}")


if __name__ == "__main__":
    main()
