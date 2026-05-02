"""Upload and run a compiled Kubeflow pipeline package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import kfp

from retrieval import config


def deploy_to_kubeflow(
    pipeline_yaml: Path,
    kubeflow_endpoint: str,
    pipeline_name: str,
    experiment_name: str,
    job_name: str,
):
    """Upload a compiled pipeline YAML and start a run."""
    client = kfp.Client(host=kubeflow_endpoint)
    pipeline = client.upload_pipeline(
        pipeline_package_path=str(pipeline_yaml),
        pipeline_name=pipeline_name,
    )
    print(f"Pipeline uploaded: {pipeline.id}")

    experiment = client.create_experiment(experiment_name)
    run = client.run_pipeline(
        experiment_id=experiment.id,
        job_name=job_name,
        pipeline_id=pipeline.id,
    )
    print(f"Pipeline run started: {run.id}")
    print(f"Monitor at: {kubeflow_endpoint}/#/runs/details/{run.id}")
    return run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-yaml", type=Path, default=config.KUBEFLOW_PIPELINE_FILE)
    parser.add_argument("--kubeflow-endpoint", required=True)
    parser.add_argument("--pipeline-name", default="MovieLens Recommender Pipeline")
    parser.add_argument("--experiment-name", default="movielens-experiments")
    parser.add_argument("--job-name", default="movielens-pipeline-run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deploy_to_kubeflow(
        pipeline_yaml=args.pipeline_yaml,
        kubeflow_endpoint=args.kubeflow_endpoint,
        pipeline_name=args.pipeline_name,
        experiment_name=args.experiment_name,
        job_name=args.job_name,
    )


if __name__ == "__main__":
    main()
