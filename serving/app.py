"""FastAPI inference service for the two-stage MovieLens recommender.

Serves retrieval (ANN over precomputed tower embeddings) + ranking (LightGBM)
with no TensorFlow at request time. The model bundle is loaded once at startup.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query

from ranking.inference import DEFAULT_BUNDLE_DIR, Recommender

BUNDLE_DIR = Path(os.environ.get("MODEL_BUNDLE_DIR", DEFAULT_BUNDLE_DIR))


@lru_cache(maxsize=1)
def get_recommender() -> Recommender:
    return Recommender.from_bundle(BUNDLE_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_recommender()  # fail fast at startup if the bundle is missing/broken
    yield


app = FastAPI(
    title="MovieLens Recommender",
    version="1.0.0",
    summary="Two-stage retrieval + ranking recommendations for MovieLens 100k users.",
    lifespan=lifespan,
)


@app.get("/healthz")
def healthz(rec: Recommender = Depends(get_recommender)) -> dict:
    return {"status": "ok", "model_version": rec.model_version, "users": len(rec.known_user_ids)}


@app.get("/model")
def model(rec: Recommender = Depends(get_recommender)) -> dict:
    return {"model_version": rec.model_version, "offline_metrics": rec.metrics}


@app.get("/users")
def users(
    rec: Recommender = Depends(get_recommender),
    limit: int = Query(50, ge=1, le=943),
) -> dict:
    ids = rec.known_user_ids
    return {"count": len(ids), "user_ids": ids[:limit]}


@app.get("/users/{user_id}/history")
def user_history(
    user_id: int,
    rec: Recommender = Depends(get_recommender),
    limit: int = Query(20, ge=1, le=100),
) -> dict:
    if not rec.has_user(user_id):
        raise HTTPException(status_code=404, detail=f"Unknown user_id {user_id}")
    frame = rec.user_history(user_id, limit=limit)
    return {
        "user_id": user_id,
        "history": [
            {
                "movie_id": int(r.movie_id),
                "title": str(r.title),
                "genres": str(r.genres),
                "rating": int(r.rating),
                "liked": bool(r.liked),
            }
            for r in frame.itertuples(index=False)
        ],
    }


@app.get("/recommend/{user_id}")
def recommend(
    user_id: int,
    rec: Recommender = Depends(get_recommender),
    k: int = Query(10, ge=1, le=100),
    n_candidates: int = Query(200, ge=10, le=1000),
    exclude_seen: bool = Query(True),
) -> dict:
    try:
        results = rec.recommend(user_id, k=k, n_candidates=n_candidates, exclude_seen=exclude_seen)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown user_id {user_id}") from None
    return {
        "user_id": user_id,
        "model_version": rec.model_version,
        "recommendations": [
            {
                "rank": i + 1,
                "movie_id": r.movie_id,
                "title": r.title,
                "genres": r.genres,
                "ranker_score": round(r.ranker_score, 6),
                "retrieval_rank": r.retrieval_rank,
            }
            for i, r in enumerate(results)
        ],
    }
