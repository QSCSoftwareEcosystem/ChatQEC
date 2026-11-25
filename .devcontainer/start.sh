#!/usr/bin/env bash
# Start the ChatQEC Streamlit app on port 8501 in the background (idempotent).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export CHATQEC_MCP_SERVER="${CHATQEC_MCP_SERVER:-chatqec-tools}"

if curl -fs -m 2 http://localhost:8501/_stcore/health >/dev/null 2>&1; then
  echo "ChatQEC is already running on port 8501."
  exit 0
fi
mkdir -p logs
# The app starts Qdrant itself if it is not running (chatqec_app/qdrant_boot.py).
nohup uv run streamlit run chatqec_app/streamlit_app.py \
  --server.port 8501 --server.headless true > logs/streamlit.log 2>&1 &
echo "ChatQEC is starting on port 8501 (log: logs/streamlit.log)."
