"""Compile the ranking Kubeflow Pipeline package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kfp import compiler

from ranking import config as ranking_config
from ranking.training.pipeline_definition import create_ranking_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=ranking_config.KUBEFLOW_RANKING_PIPELINE_FILE,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(
        pipeline_func=create_ranking_pipeline,
        package_path=str(args.output_file),
    )
    print(f"Ranking Kubeflow pipeline compiled: {args.output_file}")


if __name__ == "__main__":
    main()
