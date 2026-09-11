#!/bin/sh
# Single-container startup for hosts that expose only one public port (e.g. Render):
# run the API privately on 127.0.0.1:8000, then run Streamlit in the foreground on
# $PORT (falling back to 8501 for local `docker run`), talking to the API internally.
set -e

uvicorn serving.app:app --host 127.0.0.1 --port 8000 &
sleep 3

export API_URL="http://127.0.0.1:8000"
exec streamlit run ui/app.py \
  --server.port "${PORT:-8501}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false
