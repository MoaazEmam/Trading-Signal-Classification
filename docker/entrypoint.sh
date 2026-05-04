#!/bin/bash
if [ ! -f /app/mlflow.db ]; then
  python -c "import mlflow; mlflow.set_tracking_uri('sqlite:///mlflow.db')" || true
fi

case "$SERVICE" in
  api)
    exec uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --workers 2
    ;;
  streamlit)
    exec streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
    ;;
  scheduler)
    exec /app/scheduler/entrypoint.sh
    ;;
  *)
    echo "ERROR: SERVICE env var must be api, streamlit, or scheduler"
    exit 1
    ;;
esac