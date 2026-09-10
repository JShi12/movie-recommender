# Contributing

## Environment

Two dependency sets, because the TFX 1.14 retrieval pipeline is pinned to Python 3.9–3.10:

| Task | Install | Python |
|------|---------|--------|
| Ranking, evaluation metrics, serving API, tests | `make setup` (`pip install -e ".[dev,serve,ui]"`) | 3.9+ (3.11 recommended) |
| Running the TFX retrieval pipeline | `make setup-pipeline` (`pip install -e ".[pipeline,dev]"`) | 3.9 or 3.10 only |

## Workflow

```bash
make lint        # ruff
make format      # ruff --fix + ruff format
make typecheck   # mypy
make test        # fast unit tests, no TensorFlow needed
make test-pipeline   # slow TFX smoke tests, needs [pipeline]
```

`pre-commit install` (run by `make setup`) applies ruff and notebook-output stripping on commit.

## Conventions

- Type hints on new/edited functions; `from __future__ import annotations` at module top.
- Keep pure logic (feature math, metrics, splitting) free of TensorFlow imports so it stays unit-testable.
- New behaviour needs a test under `tests/`. Mark tests that need TFX/TensorFlow with `@pytest.mark.pipeline`.
- Don't commit regenerable artifacts (`data/`, `tfx_pipeline_output/`, `*.parquet`, `*.joblib`). The curated `serving/model_bundle/` is the one intentional exception.

## Commits

Conventional-style prefixes (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`). One logical change per commit.
