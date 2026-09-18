"""Streamlit demo UI for the MovieLens recommender API.

Run the API first (``make serve``), then ``make ui``. Set API_URL to point
elsewhere (docker-compose sets it to the ``api`` service).
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="MovieLens Recommender", page_icon="🎬", layout="wide")
st.title("🎬 MovieLens two-stage recommender")
st.caption("Retrieval (ANN over two-tower embeddings) → ranking (LightGBM LambdaRank)")


@st.cache_data(ttl=60, max_entries=100)
def api_get(path: str) -> dict:
    resp = requests.get(f"{API_URL}{path}", timeout=30)
    resp.raise_for_status()
    return resp.json()


try:
    health = api_get("/healthz")
except requests.RequestException as exc:
    st.error(f"Cannot reach the API at {API_URL} — is `make serve` running?\n\n{exc}")
    st.stop()

user_ids = api_get("/users?limit=943")["user_ids"]

with st.sidebar:
    st.header("Query")
    user_id = st.selectbox("User", user_ids, index=0)
    k = st.slider("Recommendations (k)", 5, 30, 10)
    n_candidates = st.slider("Retrieval depth", 50, 1000, 200, step=50)
    exclude_seen = st.checkbox("Exclude already-rated", value=True)
    st.divider()
    st.caption(f"Model version: `{health['model_version']}`")
    st.caption(f"Known users: {health['users']}")

hist_col, rec_col = st.columns(2)

with hist_col:
    st.subheader(f"User {user_id} — rating history")
    history = api_get(f"/users/{user_id}/history?limit=25")["history"]
    if history:
        df = pd.DataFrame(history)
        df["liked"] = df["liked"].map({True: "👍", False: "👎"})
        st.dataframe(
            df[["liked", "rating", "title", "genres"]],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No rating history for this user.")

with rec_col:
    st.subheader("Recommendations")
    payload = api_get(
        f"/recommend/{user_id}?k={k}&n_candidates={n_candidates}"
        f"&exclude_seen={str(exclude_seen).lower()}"
    )
    recs = payload["recommendations"]
    if recs:
        df = pd.DataFrame(recs)
        st.dataframe(
            df[["rank", "title", "genres", "ranker_score", "retrieval_rank"]],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No recommendations (try disabling 'exclude already-rated').")
