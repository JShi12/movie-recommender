"""Train a LightGBM LambdaRank model on prepared ranking data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import lightgbm as lgb
import pandas as pd

from ranking import config as ranking_config
from ranking.feature_builder import RANKING_FEATURES


def load_ranking_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Ranking data not found: {path}")
    return pd.read_parquet(path)


def group_counts(frame: pd.DataFrame) -> list[int]:
    return frame.sort_values("user_id").groupby("user_id").size().astype(int).tolist()


def sorted_for_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["user_id", "label", "candidate_score"], ascending=[True, False, False])


def train_ranker(
    train_path: Path = ranking_config.TRAIN_FILE,
    validation_path: Path = ranking_config.VALIDATION_FILE,
    model_file: Path = ranking_config.MODEL_FILE,
) -> None:
    train = sorted_for_ranking(load_ranking_frame(train_path))
    val = sorted_for_ranking(load_ranking_frame(validation_path))

    model = lgb.LGBMRanker(**ranking_config.LIGHTGBM_PARAMS)
    model.fit(
        train[RANKING_FEATURES],
        train["label"],
        group=group_counts(train),
        sample_weight=train["sample_weight"],
        eval_set=[(val[RANKING_FEATURES], val["label"])],
        eval_group=[group_counts(val)],
        eval_sample_weight=[val["sample_weight"]],
        eval_at=[10, 20, 100],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=25),
        ],
    )

    model_file.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_file))
    joblib.dump(model, model_file.with_suffix(".joblib"))
    ranking_config.FEATURES_FILE.write_text(
        json.dumps(RANKING_FEATURES, indent=2),
        encoding="utf-8",
    )
    print(f"Ranker saved to {model_file}")
    print(f"Joblib model saved to {model_file.with_suffix('.joblib')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=ranking_config.TRAIN_FILE)
    parser.add_argument("--validation-path", type=Path, default=ranking_config.VALIDATION_FILE)
    parser.add_argument("--model-file", type=Path, default=ranking_config.MODEL_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_ranker(args.train_path, args.validation_path, args.model_file)


if __name__ == "__main__":
    main()
