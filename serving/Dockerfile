# Serving image for the FastAPI API and the Streamlit UI (same image, two commands).
FROM python:3.11-slim AS base

# libgomp1 is the OpenMP runtime LightGBM needs at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first for layer caching.
COPY pyproject.toml README.md ./
COPY shared/ shared/
COPY retrieval/ retrieval/
COPY ranking/ ranking/
COPY serving/ serving/
COPY ui/ ui/
RUN pip install --no-cache-dir -e ".[serve,ui]"

EXPOSE 8000 8501

# Default: the API. docker-compose overrides the command for the UI service.
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
