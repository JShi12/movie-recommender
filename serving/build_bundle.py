"""Assemble serving/model_bundle/ from the latest trained artifacts.

The bundle is the small (~0.6 MB), TensorFlow-free set of files the FastAPI
service needs: tower embeddings, a freshly refitted ANN index, the LightGBM
booster, and its manifest/metrics. It is committed so ``make demo`` works on a
fresh clone with no training run.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from ranking import config as ranking_config
from retrieval import config as retrieval_config
from shared.config import PROJECT_ROOT
from shared.feature_tables import load_joined_movielens, movie_feature_table

BUNDLE_DIR = PROJECT_ROOT / "serving" / "model_bundle"
RAW_DATA_DIR = PROJECT_ROOT / "ml-100k"
MOVIE_VECTOR_COLUMNS = [f"movie_vector_{i:02d}" for i in range(32)]


def _latest_ranker_dir() -> Path:
    pushed = ranking_config.PUSHED_RANKER_DIR
    versions = (
        sorted(p for p in pushed.iterdir() if p.is_dir() and p.name.isdigit())
        if pushed.exists()
        else []
    )
    return versions[-1] if versions else ranking_config.RANKER_ARTIFACT_DIR


def build_bundle(bundle_dir: Path = BUNDLE_DIR) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tower embeddings (copied verbatim).
    for src in (
        retrieval_config.USER_EMBEDDINGS_FILE,
        retrieval_config.MOVIE_EMBEDDINGS_FILE,
    ):
        if not src.exists():
            raise FileNotFoundError(f"Missing retrieval artifact: {src}. Run `make retrieval`.")
        shutil.copy2(src, bundle_dir / src.name)

    # 2. Refit the ANN index with the installed sklearn (avoids stale-pickle warnings).
    movie_embeddings = pd.read_parquet(bundle_dir / "movie_embeddings.parquet")
    movie_vectors = movie_embeddings[MOVIE_VECTOR_COLUMNS].to_numpy()
    index = NearestNeighbors(metric="cosine", algorithm="brute")
    index.fit(movie_vectors)
    joblib.dump(
        {
            "index": index,
            "movie_ids": movie_embeddings["movie_id"].to_numpy(),
            "embedding_columns": MOVIE_VECTOR_COLUMNS,
            "metric": "cosine",
        },
        bundle_dir / "movie_ann_index.joblib",
    )

    # 3. Pre-joined history + movie table, so serving needs no raw ml-100k/.
    history = load_joined_movielens(RAW_DATA_DIR)
    history["label"] = pd.to_numeric(history["label"], errors="coerce")
    history.to_parquet(bundle_dir / "history.parquet", index=False)
    movie_feature_table(RAW_DATA_DIR).to_parquet(bundle_dir / "movies.parquet", index=False)

    # 4. LightGBM booster + metadata from the latest pushed ranker.
    ranker_dir = _latest_ranker_dir()
    for name in ("lgbm_ranker.txt", "features.json", "metrics.json", "end_to_end_metrics.json"):
        src = ranker_dir / name
        if src.exists():
            shutil.copy2(src, bundle_dir / name)

    manifest = {
        "version": ranker_dir.name if ranker_dir.name.isdigit() else "dev",
        "source_ranker_dir": str(ranker_dir.relative_to(PROJECT_ROOT)),
        "files": sorted(p.name for p in bundle_dir.iterdir() if p.is_file()),
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total = sum(p.stat().st_size for p in bundle_dir.iterdir() if p.is_file())
    print(f"Bundle written to {bundle_dir} ({total / 1024:.0f} KB)")
    for p in sorted(bundle_dir.iterdir()):
        print(f"  {p.name}")
    return bundle_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=BUNDLE_DIR)
    build_bundle(parser.parse_args().bundle_dir)


if __name__ == "__main__":
    main()
