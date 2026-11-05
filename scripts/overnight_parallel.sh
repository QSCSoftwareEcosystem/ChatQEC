#!/usr/bin/env bash
# Unattended run of the remaining local evaluations on three GPUs, meant for tmux:
#
#   tmux new -d -s chatqec-evals "scripts/overnight_parallel.sh <muse_db_run.jsonl> <gemma_db_run.jsonl>"
#   tmux attach -t chatqec-evals      # watch;  Ctrl-b d  to detach
#
# GPU 0: Muse-Glimmer (already serving; its decoder-bench resume is already running)
# GPU 1: Gemma 4, started here; Gemma's decoder-bench resume runs with its helpers on GPU 1
# GPU 2: reranker and CUDA-Q helpers of the queued runs (FTPrimitiveBench, comparison)
# When every job has finished, all three GPUs are freed. Logs: logs/evals_<time>/.
set -uo pipefail
cd "$(dirname "$0")/.."

MUSE_DB="${1:?muse decoder-bench run file}"
GEMMA_DB="${2:?gemma decoder-bench run file}"
TOOL_TIMEOUT="${TOOL_TIMEOUT:-7200}"
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs/evals_$TS"
mkdir -p "$LOGDIR"
exec > >(tee -a "$LOGDIR/main.log") 2>&1
export CHATQEC_TOOL_CACHE_DIR=data/benchmarks/tool_cache

say() { echo "[$(date '+%F %T')] $*"; }

free_gpu() {  # free_gpu <gpu> [port]: stop the vLLM server on [port], then compute processes on <gpu>
  local gpu="$1" port="${2:-}" used
  if [ -n "$port" ]; then
    for p in $(pgrep -f "vllm serve .*--port $port"); do kill "$p" 2>/dev/null || true; done
    sleep 15
  fi
  # vLLM's EngineCore worker survives its server; kill only this user's processes on <gpu>.
  for p in $(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader | tr -d ' '); do
    [ "$(ps -o user= -p "$p" 2>/dev/null)" = "$USER" ] && kill -9 "$p" 2>/dev/null
  done
  sleep 5
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
  say "GPU $gpu now at ${used} MiB"
}

say "starting Gemma on GPU 1"
GEMMA_GPU=1 GPU_MEM_UTIL=0.87 scripts/serve_local_models.sh gemma
for i in $(seq 1 120); do
  curl -s -m 2 localhost:8002/v1/models | grep -q gemma-4 && { say "Gemma ready after $((i * 10)) s"; break; }
  grep -qE "Traceback|out of memory" logs/vllm_gemma-4.log && { say "Gemma failed to start"; tail -5 logs/vllm_gemma-4.log; free_gpu 1 8002; exit 1; }
  sleep 10
done

# Job 1: Gemma's decoder-bench resume (helpers on GPU 1).
( CUDA_VISIBLE_DEVICES=1 uv run chatqec eval bench resume "$GEMMA_DB" --allow-drift \
    --tool-timeout "$TOOL_TIMEOUT" > "$LOGDIR/gemma_decoder_bench.log" 2>&1
  say "job 1 done: gemma decoder-bench, exit $?" ) &
J1=$!

# Job 2: queued new runs, one after another (helpers on GPU 2). The judge is the other model.
(
  export CUDA_VISIBLE_DEVICES=2
  run() {  # run <log name> <args...>
    local name="$1"; shift
    say "job 2 start: $name"
    uv run chatqec eval bench run "$@" --tool-timeout "$TOOL_TIMEOUT" > "$LOGDIR/$name.log" 2>&1
    say "job 2 done: $name, exit $?"
  }
  run ftpb_muse   ft-primitive-bench --model muse-glimmer --out "eval/results/ft-primitive-bench_muse-glimmer_$TS.jsonl"
  run ftpb_gemma  ft-primitive-bench --model gemma-4      --out "eval/results/ft-primitive-bench_gemma-4_$TS.jsonl"
  run cmp_chatqec_muse  compare --model muse-glimmer --system chatqec --judge-model gemma-4 --out "eval/results/compare_chatqec_muse-glimmer_$TS.jsonl"
  run cmp_bare_muse     compare --model muse-glimmer --system bare    --judge-model gemma-4 --out "eval/results/compare_bare_muse-glimmer_$TS.jsonl"
  run cmp_chatqec_gemma compare --model gemma-4      --system chatqec --judge-model muse-glimmer --out "eval/results/compare_chatqec_gemma-4_$TS.jsonl"
  run cmp_bare_gemma    compare --model gemma-4      --system bare    --judge-model muse-glimmer --out "eval/results/compare_bare_gemma-4_$TS.jsonl"
) &
J2=$!

wait "$J1" "$J2"
say "jobs 1 and 2 finished; waiting for Muse's decoder-bench resume"
while pgrep -f "chatqec.cli eval bench run decoder-bench --model muse-glimmer" > /dev/null; do sleep 60; done
say "Muse decoder-bench finished: $(($(wc -l < "$MUSE_DB") - 1)) rows"

say "freeing GPUs"
free_gpu 0 8001
free_gpu 1 8002
free_gpu 2
say "DONE. New result files: eval/results/*_$TS.jsonl; logs: $LOGDIR"
