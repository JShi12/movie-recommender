"""Bless and push a trained LightGBM ranker artifact."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ranking import config as ranking_config


def _metric(metrics: dict[str, float | int | str | None], name: str) -> float:
    value = metrics.get(name)
    if value is None:
        raise ValueError(f"Required metric is missing or null: {name}")
    return float(value)


def validate_thresholds(
    metrics: dict[str, float | int | str | None],
    min_ndcg_at_10: float,
    min_recall_at_100: float,
) -> None:
    failures = []
    ndcg_at_10 = _metric(metrics, "ndcg@10")
    recall_at_100 = _metric(metrics, "recall@100")

    if ndcg_at_10 < min_ndcg_at_10:
        failures.append(f"ndcg@10={ndcg_at_10:.6f} < {min_ndcg_at_10:.6f}")
    if recall_at_100 < min_recall_at_100:
        failures.append(f"recall@100={recall_at_100:.6f} < {min_recall_at_100:.6f}")

    if failures:
        raise RuntimeError("Ranker failed push thresholds: " + "; ".join(failures))


def copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def push_ranker(
    model_file: Path = ranking_config.MODEL_FILE,
    joblib_model_file: Path | None = None,
    features_file: Path = ranking_config.FEATURES_FILE,
    metrics_file: Path = ranking_config.METRICS_FILE,
    end_to_end_metrics_file: Path = ranking_config.END_TO_END_METRICS_FILE,
    pushed_dir: Path = ranking_config.PUSHED_RANKER_DIR,
    min_ndcg_at_10: float = ranking_config.MIN_END_TO_END_NDCG_AT_10,
    min_recall_at_100: float = ranking_config.MIN_END_TO_END_RECALL_AT_100,
    version: str | None = None,
) -> Path:
    if not model_file.exists():
        raise FileNotFoundError(f"Ranker model not found: {model_file}")
    if not end_to_end_metrics_file.exists():
        raise FileNotFoundError(f"End-to-end metrics not found: {end_to_end_metrics_file}")

    metrics = json.loads(end_to_end_metrics_file.read_text(encoding="utf-8"))
    validate_thresholds(metrics, min_ndcg_at_10, min_recall_at_100)

    joblib_model_file = joblib_model_file or model_file.with_suffix(".joblib")
    version = version or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    destination = pushed_dir / version
    destination.mkdir(parents=True, exist_ok=False)

    copy_if_exists(model_file, destination / ranking_config.PUSHED_MODEL_FILE_NAME)
    copy_if_exists(joblib_model_file, destination / ranking_config.PUSHED_JOBLIB_FILE_NAME)
    copy_if_exists(features_file, destination / ranking_config.PUSHED_FEATURES_FILE_NAME)
    copy_if_exists(metrics_file, destination / ranking_config.PUSHED_METRICS_FILE_NAME)
    copy_if_exists(
        end_to_end_metrics_file,
        destination / ranking_config.PUSHED_END_TO_END_METRICS_FILE_NAME,
    )

    manifest = {
        "version": version,
        "pushed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_file": ranking_config.PUSHED_MODEL_FILE_NAME,
        "joblib_model_file": (
            ranking_config.PUSHED_JOBLIB_FILE_NAME if joblib_model_file.exists() else None
        ),
        "features_file": (
            ranking_config.PUSHED_FEATURES_FILE_NAME if features_file.exists() else None
        ),
        "metrics_file": (
            ranking_config.PUSHED_METRICS_FILE_NAME if metrics_file.exists() else None
        ),
        "end_to_end_metrics_file": ranking_config.PUSHED_END_TO_END_METRICS_FILE_NAME,
        "thresholds": {
            "min_ndcg@10": min_ndcg_at_10,
            "min_recall@100": min_recall_at_100,
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Ranker pushed to {destination}")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-file", type=Path, default=ranking_config.MODEL_FILE)
    parser.add_argument("--joblib-model-file", type=Path, default=None)
    parser.add_argument("--features-file", type=Path, default=ranking_config.FEATURES_FILE)
    parser.add_argument("--metrics-file", type=Path, default=ranking_config.METRICS_FILE)
    parser.add_argument(
        "--end-to-end-metrics-file",
        type=Path,
        default=ranking_config.END_TO_END_METRICS_FILE,
    )
    parser.add_argument("--pushed-dir", type=Path, default=ranking_config.PUSHED_RANKER_DIR)
    parser.add_argument(
        "--min-ndcg-at-10",
        type=float,
        default=ranking_config.MIN_END_TO_END_NDCG_AT_10,
    )
    parser.add_argument(
        "--min-recall-at-100",
        type=float,
        default=ranking_config.MIN_END_TO_END_RECALL_AT_100,
    )
    parser.add_argument("--version", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    push_ranker(
        model_file=args.model_file,
        joblib_model_file=args.joblib_model_file,
        features_file=args.features_file,
        metrics_file=args.metrics_file,
        end_to_end_metrics_file=args.end_to_end_metrics_file,
        pushed_dir=args.pushed_dir,
        min_ndcg_at_10=args.min_ndcg_at_10,
        min_recall_at_100=args.min_recall_at_100,
        version=args.version,
    )


if __name__ == "__main__":
    main()
