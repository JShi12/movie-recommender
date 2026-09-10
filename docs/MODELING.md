# Modelling notes

## Data and splits

- MovieLens 100k: 100,000 ratings (1–5), 943 users, 1,682 movies, Sep 1997 – Apr 1998.
- **Chronological** split 80 / 10 / 10 by timestamp (`shared.movielens.time_based_split`).
  The boundary is snapped to a timestamp-group edge, so a single timestamp never
  appears in two partitions — no look-ahead leakage in evaluation.
- Retrieval labels: rating ≥ 4 → positive, rating ≤ 2 → negative, rating = 3 dropped
  (`NEUTRAL_RATING`) for a cleaner signal.
- Ranking relevance grades (`rating_to_relevance`): 5 → 3, 4 → 2, 3 → 1, else 0.
- Sample weights: observed interactions 1.0, unobserved candidates 0.3.

## Leakage control

Every behavioural feature is a **pre-event** quantity. `add_historical_observed_features`
sorts by `(timestamp, user_id, movie_id)` and builds cumulative stats with the current
row removed:

```text
user_rating_count_before   = cumcount()                       # 0 for a user's first row
user_avg_rating_before      = (cumsum(rating) - rating) / count_before
user_like_rate_before       = (cumsum(is_like) - is_like) / count_before
user_activity_gap_log       = log1p(timestamp - previous_user_timestamp)
… and the movie_* equivalents
```

Unseen users/movies fall back to global means. For ranking *candidates* (no observed
rating) `fill_candidate_historical_features` attaches each user's / movie's **last
known** value from history. Genre affinity (`add_genre_affinity`) is the dot product of
the user's historical liked-genre distribution with the candidate movie's genre
one-hot.

## Feature vector (`RANKING_FEATURES`, 162 columns)

| Group | Count | Examples |
|---|---:|---|
| Retrieval signal | 2 | `candidate_score`, `retrieval_rank` |
| User history | 4 | `user_avg_rating_before`, `user_rating_count_before`, `user_like_rate_before`, `user_activity_gap_log` |
| Movie history | 4 | `movie_avg_rating_before`, `movie_rating_count_before`, `movie_like_rate_before`, `movie_popularity_before` |
| Context / cross | 5 | `release_year`, `hour_of_day`, `day_of_week`, `user_avg_minus_movie_avg`, `user_genre_affinity` |
| Genre one-hot | 19 | `genre_Action`, `genre_Comedy`, … |
| Retrieval vectors | 4 × 32 | `user_retrieval_vector_*`, `movie_retrieval_vector_*`, their element-wise product and \|difference\| |

Serving builds this exact vector from the same code (`ranking.features`), guaranteeing
train/serve parity.

## Hyperparameters

**Retrieval** (`retrieval/config.py`): embeddings — user/movie 32, age 8, gender 4,
occupation 12, genre 12, final 32; dropout 0.20; L2 1e-6; Adam 1e-3; batch 64; up to
10 epochs, 1,500 train steps; `EarlyStopping(val_auc, patience=1)`.

**Ranking** (`ranking/config.py` `LIGHTGBM_PARAMS`): `lambdarank` / `ndcg`, 500
estimators, learning rate 0.05, `num_leaves` 30, `min_child_samples` 40, `subsample`
0.7, `colsample_bytree` 0.8, `reg_alpha` 0.5, `reg_lambda` 1.0; early stopping 50
rounds on validation nDCG; one LightGBM group per user.

## Metrics

Evaluated over users with ≥ 1 held-out positive (153 on the test split).

| Metric | Definition |
|---|---|
| `recall@k` | mean over users of (held-out positives in top-k) / (held-out positives) |
| `micro_recall@k` | Σ hits / Σ positives (pooled, not per-user averaged) |
| `hit_rate@k` | fraction of users with ≥ 1 positive in top-k |
| `nDCG@k` | binary gains, ideal DCG uses min(k, #positives) |
| `candidate_coverage@k` | distinct movies ever recommended / catalogue size |
| `candidate_auc` | ROC-AUC over all candidate rows, label = is-held-out-positive |
| `mean_positive_rank@k` | average retrieval rank of recovered positives |

Retrieval optimises recall/hit-rate at large k (get the item into the set);
end-to-end optimises nDCG/recall at small k (put it near the top). See the tables in
the [README](../README.md#results).

## Known limitations

- **Cold start:** serving only handles users present in `user_embeddings.parquet`.
  A production system would run the user tower online (or cache it) for new users.
- Popularity bias: `movie_popularity_before` and history features favour well-rated
  catalogue staples; no explicit diversity or exploration term.
- Single train/eval run — no cross-validation or confidence intervals.
