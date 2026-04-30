"""Evaluate the trained LightGBM ranker on prepared ranking data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

from ranking import config as ranking_config
from ranking.feature_builder import RANKING_FEATURES, ndcg_at_k, recall_at_k


def evaluate_ranker(
    data_path: Path = ranking_config.TEST_FILE,
    model_file: Path = ranking_config.MODEL_FILE,
) -> dict[str, float]:
    frame = pd.read_parquet(data_path).sort_values("user_id")
    booster = lgb.Booster(model_file=str(model_file))
    frame["ranker_score"] = booster.predict(frame[RANKING_FEATURES])

    metrics = {
        "auc": float(roc_auc_score(frame["label"], frame["ranker_score"])),
    }
    for k in (10, 20, 100):
        ndcgs = []
        recalls = []
        for _, group in frame.groupby("user_id"):
            labels = group["label"].astype(float).tolist()
            scores = group["ranker_score"].astype(float).tolist()
            ndcgs.append(ndcg_at_k(labels, scores, k))
            recalls.append(recall_at_k(labels, scores, k))
        metrics[f"ndcg@{k}"] = float(sum(ndcgs) / len(ndcgs))
        metrics[f"recall@{k}"] = float(sum(recalls) / len(recalls))

    ranking_config.METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    ranking_config.METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {ranking_config.METRICS_FILE}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=ranking_config.TEST_FILE)
    parser.add_argument("--model-file", type=Path, default=ranking_config.MODEL_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_ranker(args.data_path, args.model_file)


if __name__ == "__main__":
    main()
