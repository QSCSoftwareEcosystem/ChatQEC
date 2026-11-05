#!/usr/bin/env bash
# Unattended single-GPU chain: finish the Muse benchmark run, swap Gemma onto the same GPU,
# resume Gemma's run, then free the GPU. Meant to run inside tmux so it survives logout:
#
#   tmux new -d -s chatqec-overnight "scripts/overnight_single_gpu.sh <muse_run.jsonl> <gemma_run.jsonl>"
#   tmux attach -t chatqec-overnight        # watch;  Ctrl-b d  to detach
#
# Every step is logged to logs/overnight_<time>.log. Only GPU $GPU is used; only processes on
# that GPU are stopped.
set -uo pipefail
cd "$(dirname "$0")/.."

MUSE_RUN="${1:?muse run file}"
GEMMA_RUN="${2:?gemma run file}"
GPU="${GPU:-0}"
TOOL_TIMEOUT="${TOOL_TIMEOUT:-7200}"
LOG="logs/overnight_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
exec > >(tee -a "$LOG") 2>&1

say() { echo "[$(date '+%F %T')] $*"; }
rows() { echo $(($(wc -l < "$1") - 1)); }

free_gpu() {  # free_gpu <port>: stop the vLLM server on <port>, then any compute process left on $GPU
  local port="$1" used
  for p in $(pgrep -f "vllm serve .*--port $port"); do kill "$p" 2>/dev/null || true; done
  sleep 15
  # vLLM's EngineCore worker survives its server; it is a compute process on $GPU.
  for p in $(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader | tr -d ' '); do
    kill -9 "$p" 2>/dev/null || true
  done
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$used" -lt 1000 ] && { say "GPU $GPU free (${used} MiB)"; return 0; }
    sleep 5
  done
  say "WARNING: GPU $GPU still at ${used} MiB"
}

say "waiting for the Muse run to finish ($(rows "$MUSE_RUN")/40 rows so far)"
while pgrep -f "chatqec.cli eval bench run decoder-bench --model muse-glimmer" > /dev/null; do sleep 60; done
say "Muse run finished: $(rows "$MUSE_RUN") rows"

say "stopping Muse on GPU $GPU"
free_gpu 8001

say "starting Gemma on GPU $GPU"
GEMMA_GPU="$GPU" GPU_MEM_UTIL=0.87 scripts/serve_local_models.sh gemma
for i in $(seq 1 120); do
  curl -s -m 2 localhost:8002/v1/models | grep -q gemma-4 && { say "Gemma ready after $((i * 10)) s"; break; }
  grep -qE "Traceback|out of memory" logs/vllm_gemma-4.log && { say "Gemma failed to start"; tail -5 logs/vllm_gemma-4.log; free_gpu 8002; exit 1; }
  sleep 10
done

say "resuming the Gemma run ($(rows "$GEMMA_RUN")/40 rows so far)"
CUDA_VISIBLE_DEVICES="$GPU" CHATQEC_TOOL_CACHE_DIR=data/benchmarks/tool_cache \
  uv run chatqec eval bench resume "$GEMMA_RUN" --allow-drift --tool-timeout "$TOOL_TIMEOUT"
say "Gemma run finished: $(rows "$GEMMA_RUN") rows"

say "stopping Gemma and freeing GPU $GPU"
free_gpu 8002
say "DONE. Results: $MUSE_RUN, $GEMMA_RUN. Resume notes: logs/bench_resume_notes.txt"
