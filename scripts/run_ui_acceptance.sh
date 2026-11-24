#!/usr/bin/env bash
# Run the Playwright UI acceptance prompts against both local models through the
# real Streamlit Chat page, one app per model, each pinned to its own GPU.
#
#   scripts/run_ui_acceptance.sh <out_dir>
#
# Needs the model servers already up (Muse :8001 on GPU 0, Gemma :8002 on GPU 1):
#   GPU_MEM_UTIL=0.80 scripts/serve_local_models.sh muse
#   GPU_MEM_UTIL=0.87 scripts/serve_local_models.sh gemma
# The lower GPU share leaves room on each GPU for the reranker and CUDA-Q decoders.
set -euo pipefail

OUT="${1:?usage: $0 <out_dir>}"
PROMPTS="${PROMPTS:-eval/ui_acceptance_prompts.json}"
CHROME="${CHROME:-$HOME/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome}"
cd "$(dirname "$0")/.."
mkdir -p "$OUT"

pids=()
cleanup() {
  for p in "${pids[@]}"; do pkill -P "$p" 2>/dev/null || true; kill "$p" 2>/dev/null || true; done
}
trap cleanup EXIT

start_app() {  # port model gpu
  CUDA_VISIBLE_DEVICES="$3" CHATQEC_PROVIDER=local CHATQEC_LOCAL_MODEL="$2" \
    CHATQEC_MCP_SERVER=chatqec-tools \
    uv run streamlit run chatqec_app/streamlit_app.py --server.port "$1" --server.headless true \
    > "$OUT/streamlit_$2.log" 2>&1 &
  pids+=("$!")
}
start_app 8501 muse-glimmer 0
start_app 8502 gemma-4 1

for _ in $(seq 1 60); do
  [ "$(curl -s -m 2 localhost:8501/_stcore/health)$(curl -s -m 2 localhost:8502/_stcore/health)" = "okok" ] && break
  sleep 2
done

run() {  # port label
  uv run --with playwright python scripts/ui_acceptance.py --port "$1" --label "$2" \
    --prompts "$PROMPTS" --out "$OUT" --executable-path "$CHROME" > "$OUT/run_$2.log" 2>&1
}
run 8501 muse & r1=$!
run 8502 gemma & r2=$!
wait "$r1" "$r2"

uv run python - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
for label in ("muse", "gemma"):
    rows = json.loads((out / f"ui_{label}.json").read_text())
    hit = sum(set(r["expected_tools"]) <= set(r["tools_shown"]) for r in rows if r["expected_tools"])
    need = sum(bool(r["expected_tools"]) for r in rows)
    spurious = sum(bool(r["tools_shown"]) for r in rows if not r["expected_tools"])
    ctrl = sum(not r["expected_tools"] for r in rows)
    errs = sum(r["exceptions"] > 0 or r["status"] != "ok" for r in rows)
    print(f"{label}: expected tool called {hit}/{need}, tool on no-tool prompt {spurious}/{ctrl}, errors {errs}")
EOF
