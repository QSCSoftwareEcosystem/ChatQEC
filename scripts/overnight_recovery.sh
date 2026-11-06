#!/usr/bin/env bash
# Unattended recovery of the 2026-11-05 evaluation runs, for tmux (survives logout):
#
#   tmux new -d -s chatqec-recovery scripts/overnight_recovery.sh
#   tail -f logs/recovery_<time>/main.log
#
# Expects Muse on GPU 0 (:8001) and Gemma on GPU 1 (:8002) already serving. Three lanes run in
# parallel, each with its helper processes (reranker, CUDA-Q) on its own GPU:
#   lane 0 (GPU 0): Muse decoder-bench resume, then Muse FTPrimitiveBench resume (retry errors)
#   lane 1 (GPU 1): Gemma decoder-bench resume, then a new Gemma FTPrimitiveBench run
#   lane 2 (GPU 2): the four ChatQEC-vs-bare comparison runs (judge = the other model)
# Seeded tool results come from the shared cache, so lost rows are re-asked cheaply.
# When all lanes are done, this user's processes on GPUs 0-2 are stopped.
# Run outputs are gitignored: do not commit while this runs (see .gitignore).
set -uo pipefail
cd "$(dirname "$0")/.."

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs/recovery_$TS"
mkdir -p "$LOGDIR"
exec > >(tee -a "$LOGDIR/main.log") 2>&1
export CHATQEC_TOOL_CACHE_DIR=data/benchmarks/tool_cache
TIMEOUT=7200
R=eval/results

say() { echo "[$(date '+%F %T')] $*"; }
rows() { [ -f "$1" ] && echo $(($(wc -l < "$1") - 1)) || echo 0; }

for port in 8001 8002; do
  curl -s -m 5 "localhost:$port/v1/models" | grep -q '"id"' || { say "no model on :$port; aborting"; exit 1; }
done

step() {  # step <lane> <log name> <command...>
  local lane="$1" name="$2"; shift 2
  say "lane $lane start: $name"
  "$@" > "$LOGDIR/$name.log" 2>&1
  say "lane $lane done: $name (exit $?)"
}

(
  export CUDA_VISIBLE_DEVICES=0
  step 0 db_muse uv run chatqec eval bench resume "$R/decoder-bench_muse-glimmer_20261104_173715.jsonl" \
    --allow-drift --tool-timeout "$TIMEOUT"
  step 0 ftpb_muse uv run chatqec eval bench resume "$R/ft-primitive-bench_muse-glimmer_20261105_153626.jsonl" \
    --allow-drift --retry-errors --tool-timeout "$TIMEOUT"
) &
L0=$!
(
  export CUDA_VISIBLE_DEVICES=1
  step 1 db_gemma uv run chatqec eval bench resume "$R/decoder-bench_gemma-4_20261104_173715.jsonl" \
    --allow-drift --tool-timeout "$TIMEOUT"
  step 1 ftpb_gemma uv run chatqec eval bench run ft-primitive-bench --model gemma-4 \
    --out "$R/ft-primitive-bench_gemma-4_$TS.jsonl" --tool-timeout "$TIMEOUT"
) &
L1=$!
(
  export CUDA_VISIBLE_DEVICES=2
  for spec in "muse-glimmer chatqec gemma-4" "muse-glimmer bare gemma-4" \
              "gemma-4 chatqec muse-glimmer" "gemma-4 bare muse-glimmer"; do
    set -- $spec
    step 2 "cmp_$2_$1" uv run chatqec eval bench run compare --model "$1" --system "$2" \
      --judge-model "$3" --out "$R/compare_$2_$1_$TS.jsonl" --tool-timeout "$TIMEOUT"
  done
) &
L2=$!

wait "$L0" "$L1" "$L2"
say "all lanes done. rows: db_muse $(rows "$R/decoder-bench_muse-glimmer_20261104_173715.jsonl")/40," \
    "db_gemma $(rows "$R/decoder-bench_gemma-4_20261104_173715.jsonl")/40," \
    "ftpb_muse $(rows "$R/ft-primitive-bench_muse-glimmer_20261105_153626.jsonl")/17," \
    "ftpb_gemma $(rows "$R/ft-primitive-bench_gemma-4_$TS.jsonl")/17"

say "freeing GPUs 0-2"
for port in 8001 8002; do
  for p in $(pgrep -f "vllm serve .*--port $port"); do kill "$p" 2>/dev/null || true; done
done
sleep 15
for gpu in 0 1 2; do
  for p in $(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader | tr -d ' '); do
    [ "$(ps -o user= -p "$p" 2>/dev/null)" = "$USER" ] && kill -9 "$p" 2>/dev/null
  done
done
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
say "DONE. Logs: $LOGDIR"
