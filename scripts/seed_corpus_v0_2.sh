#!/usr/bin/env bash
# Seed the ChatQEC v0.2 corpus: ~18 foundational QEC sources.
#
# Phase 1 only — produces JSONL chunks + pending_review registry rows.
# Run Phase 2 (Qdrant + embed) separately after inspecting the output.
#
# Usage:
#   bash scripts/seed_corpus_v0_2.sh           # ingest everything
#   bash scripts/seed_corpus_v0_2.sh --dry     # echo commands only
#
# Re-runnable: each ingest is idempotent on source_id.

set -euo pipefail

DRY=${1:-}
run() {
  if [ "$DRY" = "--dry" ]; then
    echo "+ $*"
  else
    echo ">>> $*"
    "$@" || echo "  (skipped — see warning above)"
  fi
}

cd "$(dirname "$0")/.."

# -----------------------------------------------------------------------------
# arXiv papers — 13 foundational + recent (verified IDs)
# -----------------------------------------------------------------------------

# Foundational
run uv run chatqec ingest --arxiv quant-ph/9707021  # Kitaev — anyons / toric code
run uv run chatqec ingest --arxiv quant-ph/9601029  # Steane — [[7,1,3]] code
run uv run chatqec ingest --arxiv quant-ph/9906129  # Aharonov-Ben-Or — threshold theorem
run uv run chatqec ingest --arxiv quant-ph/0403025  # Bravyi-Kitaev — magic state distillation
run uv run chatqec ingest --arxiv quant-ph/0008040  # GKP — qubit in an oscillator

# Surface code & lattice surgery
run uv run chatqec ingest --arxiv 1208.0928         # Fowler — surface codes for FTQC (review)
run uv run chatqec ingest --arxiv 1808.02892        # Litinski — Game of Surface Codes

# Recent breakthroughs
run uv run chatqec ingest --arxiv 2408.13687        # Google Willow — below threshold (Acharya 2024)
run uv run chatqec ingest --arxiv 2308.07915        # IBM bivariate bicycle qLDPC (Bravyi 2024)
run uv run chatqec ingest --arxiv 2409.17595        # Magic state cultivation (Gidney/Litinski 2024)
run uv run chatqec ingest --arxiv 2107.02194        # Hastings-Haah honeycomb / Floquet code
run uv run chatqec ingest --arxiv 2201.03540        # Erasure conversion in neutral atoms (Wu 2022)
run uv run chatqec ingest --arxiv 2101.09310        # Fusion-based quantum computation (PsiQuantum)

# -----------------------------------------------------------------------------
# Web sources — 5 high-quality blogs / labs
# -----------------------------------------------------------------------------

run uv run chatqec ingest --url "https://errorcorrectionzoo.org/c/surface"
run uv run chatqec ingest --url "https://errorcorrectionzoo.org/c/stabilizer"
run uv run chatqec ingest --url "https://errorcorrectionzoo.org/c/quantum_reed_muller"  # [[15,1,3]] — foundation of magic state distillation
run uv run chatqec ingest --url "https://errorcorrectionzoo.org/c/css"
run uv run chatqec ingest --url "https://errorcorrectionzoo.org/c/qldpc"

echo
echo "=== seed complete ==="
echo "inspect registry: uv run chatqec sources list"
echo "inspect chunks:   ls -1 data/chunks/ | head"
echo "embed phase:      see scripts/embed_corpus_v0_2.sh"
