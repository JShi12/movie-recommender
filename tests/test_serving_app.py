"""Tests for the FastAPI serving layer (routes, shapes, error codes)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="the [serve] extra (fastapi) is not installed")

from fastapi.testclient import TestClient  # noqa: E402

from serving.app import app, get_recommender  # noqa: E402


@pytest.fixture
def client(recommender):
    # No `with` block: TestClient runs the lifespan only as a context manager, and
    # we want to skip the real bundle load and inject the fixture instead.
    app.dependency_overrides[get_recommender] = lambda: recommender
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["users"] == 943


def test_model_endpoint_reports_version(client):
    resp = client.get("/model")
    assert resp.status_code == 200
    assert "model_version" in resp.json()


def test_users_listing_is_capped_by_limit(client):
    body = client.get("/users?limit=5").json()
    assert body["count"] == 943
    assert body["user_ids"] == [1, 2, 3, 4, 5]


class TestHistoryRoute:
    def test_ok(self, client):
        body = client.get("/users/1/history?limit=3").json()
        assert body["user_id"] == 1
        assert 1 <= len(body["history"]) <= 3
        assert {"movie_id", "title", "genres", "rating", "liked"} == set(body["history"][0])

    def test_unknown_user_is_404(self, client):
        assert client.get("/users/99999/history").status_code == 404


class TestRecommendRoute:
    def test_shape_and_ordering(self, client):
        body = client.get("/recommend/1?k=8&n_candidates=200").json()
        recs = body["recommendations"]
        assert len(recs) == 8
        assert [r["rank"] for r in recs] == list(range(1, 9))
        assert {"movie_id", "title", "genres", "ranker_score", "retrieval_rank"} <= set(recs[0])

    def test_unknown_user_is_404(self, client):
        assert client.get("/recommend/99999").status_code == 404

    def test_k_out_of_range_is_422(self, client):
        assert client.get("/recommend/1?k=0").status_code == 422
