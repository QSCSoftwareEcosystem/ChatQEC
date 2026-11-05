#!/usr/bin/env bash
# Resume every unfinished benchmark run in eval/results/, detached from this terminal and
# from any Claude session, so closing either does not stop it.
#
#   scripts/resume_bench.sh                 # resume what is missing
#   scripts/resume_bench.sh --retry-errors  # also re-ask items that errored (e.g. timeouts)
#   scripts/resume_bench.sh --dry-run       # only report each run's state (runs in the foreground)
#
# Extra arguments go to `chatqec eval bench resume` (see --help). Runs whose model is not
# being served are reported as WAIT and skipped: serve the model on the one GPU first
# (scripts/serve_local_models.sh), then run this again. Output: logs/resume_bench_<time>.log
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CHATQEC_TOOL_CACHE_DIR="${CHATQEC_TOOL_CACHE_DIR:-data/benchmarks/tool_cache}"

for arg in "$@"; do
  if [ "$arg" = "--dry-run" ]; then
    exec uv run chatqec eval bench resume "$@"
  fi
done

mkdir -p logs
log="logs/resume_bench_$(date +%Y%m%d_%H%M%S).log"
setsid nohup uv run chatqec eval bench resume "$@" > "$log" 2>&1 < /dev/null &
echo "resuming in the background (pid $!); log: $log"
