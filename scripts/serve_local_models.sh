#!/usr/bin/env bash
# Serve the local test LLMs behind OpenAI-compatible endpoints for models.provider=local.
#   Muse-Glimmer-30B (primary)  -> http://localhost:8001/v1  (GPU 0)
#   Gemma 4 31B-it   (fallback) -> http://localhost:8002/v1  (GPU 1)
# FP8 weights (~31 GB) keep each model on a single 48 GB GPU.
# vLLM lives in its own venv so its pins never touch chatQEC's environment:
#   uv venv ~/venvs/llmserve --python 3.12 && VIRTUAL_ENV=~/venvs/llmserve uv pip install vllm
# Usage: scripts/serve_local_models.sh [muse|gemma|all]   (default: all; logs in logs/)
set -euo pipefail

VLLM="${VLLM_BIN:-$HOME/venvs/llmserve/bin/vllm}"
export PATH="$(dirname "$VLLM"):$PATH"  # vLLM JIT-compiles kernels with the venv's ninja
MODELS_DIR="${LOCAL_MODELS_DIR:-$HOME/local_models}"
LOG_DIR="$(dirname "$0")/../logs"
# Fraction of each GPU vLLM may take. 0.90 leaves ~5 GB, too little when the
# reranker and CUDA-Q decoders share the GPU; use 0.80 for UI/eval runs.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
# MUSE_GPU / GEMMA_GPU pick the GPU per model (defaults 0 / 1). On a shared box run
# one model at a time on one GPU, e.g. GEMMA_GPU=0 GPU_MEM_UTIL=0.87 ... gemma
mkdir -p "$LOG_DIR"

serve() {  # gpu port served_name model_dir tool_parser reasoning_parser
  CUDA_VISIBLE_DEVICES="$1" nohup "$VLLM" serve "$MODELS_DIR/$4" \
    --served-model-name "$3" --port "$2" --quantization fp8_per_tensor \
    --max-model-len 32768 --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enable-auto-tool-choice --tool-call-parser "$5" --reasoning-parser "$6" \
    > "$LOG_DIR/vllm_$3.log" 2>&1 &
  echo "$3: pid $! on GPU $1, port $2 (log: $LOG_DIR/vllm_$3.log)"
}

case "${1:-all}" in
  muse) serve "${MUSE_GPU:-0}" 8001 muse-glimmer Muse-Glimmer-30B muse_glimmer muse_glimmer ;;
  gemma) serve "${GEMMA_GPU:-1}" 8002 gemma-4 gemma-4-31B-it gemma4 gemma4 ;;
  all)
    serve "${MUSE_GPU:-0}" 8001 muse-glimmer Muse-Glimmer-30B muse_glimmer muse_glimmer
    serve "${GEMMA_GPU:-1}" 8002 gemma-4 gemma-4-31B-it gemma4 gemma4
    ;;
  *) echo "usage: $0 [muse|gemma|all]" >&2; exit 2 ;;
esac
