#!/usr/bin/env bash
# Unattended capture of the paper's Figure 2 screenshots on GPU 0 only (for tmux):
#   tmux new -d -s chatqec-fig2 scripts/capture_figure2.sh
# Serves Muse on GPU 0, starts the Chat app, captures the seeded computation and the refusal
# scenarios into 6ac52ba22379cb285b6dc6d7/figures/screens_final/, then stops everything it
# started and frees GPU 0. Writes no tracked files, so committing while it runs is safe.
set -uo pipefail
cd "$(dirname "$0")/.."
TS="$(date +%Y%m%d_%H%M%S)"; LOG="logs/fig2_$TS"; mkdir -p "$LOG"
exec > >(tee -a "$LOG/main.log") 2>&1
say() { echo "[$(date '+%F %T')] $*"; }
OUT=6ac52ba22379cb285b6dc6d7/figures/screens_final
export CHATQEC_TOOL_CACHE_DIR=data/benchmarks/tool_cache

say "serving Muse on GPU 0"
MUSE_GPU=0 GPU_MEM_UTIL=0.80 scripts/serve_local_models.sh muse > "$LOG/serve.log" 2>&1
for i in $(seq 1 90); do curl -s -m 3 localhost:8001/v1/models | grep -q '"id"' && break; sleep 10; done
curl -s -m 3 localhost:8001/v1/models | grep -q '"id"' || { say "Muse did not come up"; }

say "starting app on :8503"
CUDA_VISIBLE_DEVICES=0 CHATQEC_PROVIDER=local CHATQEC_LOCAL_MODEL=muse-glimmer \
  CHATQEC_MCP_SERVER=chatqec-mcp-tools setsid uv run streamlit run chatqec_app/streamlit_app.py \
  --server.port 8503 --server.headless true > "$LOG/streamlit.log" 2>&1 &
APP=$!
for i in $(seq 1 60); do [ "$(curl -s -m 2 localhost:8503/_stcore/health)" = ok ] && break; sleep 5; done

say "capturing"
uv run --with playwright python scripts/capture_screenshots.py --port 8503 --out "$OUT" \
  --only computation refusal \
  --executable-path ~/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome > "$LOG/capture.log" 2>&1
say "capture exit $?"; ls -la "$OUT"

say "stopping app and Muse, freeing GPU 0"
kill -- -"$APP" 2>/dev/null || kill "$APP" 2>/dev/null
for p in $(pgrep -f "vllm serve .*--port 8001"); do kill "$p" 2>/dev/null; done
sleep 15
for p in $(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader | tr -d ' '); do
  [ "$(ps -o user= -p "$p" 2>/dev/null)" = "$USER" ] && kill -9 "$p" 2>/dev/null
done
sleep 5; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
say "DONE. Logs: $LOG"
