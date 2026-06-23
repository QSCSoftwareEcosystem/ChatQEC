#!/usr/bin/env bash
# Phase 2: bring up Qdrant, embed all pending_review sources, promote to active.
#
# Prereqs:
#   - Docker + docker-compose installed
#   - GEMINI_API_KEY set in env or .env
#   - Phase 1 (seed_corpus_v0_2.sh) already ran successfully
#
# Usage:
#   bash scripts/embed_corpus_v0_2.sh

set -euo pipefail

cd "$(dirname "$0")/.."

# Sanity check
if [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "ERROR: GEMINI_API_KEY not set. export it or add to .env first."
  exit 1
fi

n_pending=$(uv run chatqec sources list --status pending_review 2>&1 | tail -1 | awk '{print $1}')
if [ "${n_pending:-0}" = "0" ]; then
  echo "no pending_review sources — run seed_corpus_v0_2.sh first."
  exit 0
fi
echo "$n_pending pending_review sources ready to embed."

# 1. Make sure Qdrant is running (we use the local binary launcher, not Docker)
echo ">>> checking Qdrant..."
if ! curl -sf http://localhost:6333/healthz > /dev/null 2>&1; then
  echo "    Qdrant not running. Start it in another terminal with:"
  echo "      bash scripts/qdrant_local.sh"
  echo "    or in the background with:"
  echo "      nohup bash scripts/qdrant_local.sh > /tmp/qdrant.log 2>&1 &"
  exit 2
fi
echo "    ready."

# 2. Embed every pending_review source
echo ">>> embedding all pending_review sources (Gemini text-embedding-004)..."
uv run chatqec embed --all-pending --provider gemini

# 3. Verify
echo ">>> sources status after embed:"
uv run chatqec sources list

echo
echo "=== embed complete ==="
echo "try a query:"
echo "  uv run chatqec ask \"What is the surface code threshold?\""
echo "  uv run chatqec ask \"Cite the original GKP code paper.\""
echo "  uv run chatqec ask \"What is magic state cultivation?\""
