#!/usr/bin/env bash
# Seed the ChatQEC v0.4 corpus: more fault-tolerance / QEC papers + QEC vendor sites.
#
# Adds on top of v0.3 (arXiv + book chapters + errorcorrectionzoo):
#   - Foundational FT/QEC papers not yet in the corpus (CSS, Knill-Laflamme,
#     topological memory, color codes, magic-state overhead, qLDPC).
#   - Landmark hardware fault-tolerance demonstrations (Google, Quantinuum,
#     Harvard/QuEra).
#   - Important QEC vendor / lab pages (error-correction programs & roadmaps).
#
# Papers use `--resolve-by-title` (the CLI's preferred, ID-error-proof path) with
# `--full-text --describe-figures` for full ingestion incl. vision-described figures.
# Vendor pages use `--url` (HTML → blog tier).
#
# Phase 1 only — produces JSONL chunks + pending_review registry rows.
# Run `chatqec embed --all-pending` (Phase 2) separately after inspecting output.
#
# Usage:
#   bash scripts/seed_corpus_v0_4.sh          # ingest everything
#   bash scripts/seed_corpus_v0_4.sh --dry    # echo commands only
#
# Re-runnable: each ingest is idempotent on source_id.

set -uo pipefail   # not -e: one bad fetch must not abort the batch

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
# Foundational fault-tolerance / QEC papers (full LaTeX + figures)
# Explicit arXiv ids (verified titles); --full-text needs --arxiv, not --resolve-by-title.
# -----------------------------------------------------------------------------
run uv run chatqec ingest --arxiv quant-ph/9512032 --full-text --describe-figures  # Calderbank-Shor: Good QEC codes exist (CSS)
run uv run chatqec ingest --arxiv quant-ph/9604034 --full-text --describe-figures  # Knill-Laflamme: theory of QEC codes (KL conditions)
run uv run chatqec ingest --arxiv quant-ph/0110143 --full-text --describe-figures  # Dennis-Kitaev-Landahl-Preskill: topological quantum memory
run uv run chatqec ingest --arxiv quant-ph/0605138 --full-text --describe-figures  # Bombin-Martin-Delgado: topological quantum distillation (color codes)
run uv run chatqec ingest --arxiv 1209.2426 --full-text --describe-figures         # Bravyi-Haah: magic state distillation with low overhead
run uv run chatqec ingest --arxiv 1905.06903 --full-text --describe-figures        # Litinski: magic state distillation not as costly
run uv run chatqec ingest --arxiv 1310.2984 --full-text --describe-figures         # Gottesman: FT computation with constant overhead
run uv run chatqec ingest --arxiv 2111.03654 --full-text --describe-figures        # Panteleev-Kalachev: asymptotically good qLDPC

# -----------------------------------------------------------------------------
# Landmark hardware fault-tolerance demonstrations (full LaTeX + figures)
# -----------------------------------------------------------------------------
run uv run chatqec ingest --arxiv 2207.06431 --full-text --describe-figures        # Google Quantum AI: suppressing errors by scaling a surface-code logical qubit
run uv run chatqec ingest --arxiv 2107.07505 --full-text --describe-figures        # Ryan-Anderson (Quantinuum): real-time fault-tolerant QEC
run uv run chatqec ingest --arxiv 2312.03982 --full-text --describe-figures        # Bluvstein (Harvard/QuEra): logical processor on reconfigurable atom arrays

# -----------------------------------------------------------------------------
# Important QEC vendor / lab pages (HTML → blog tier; URLs verified reachable)
# -----------------------------------------------------------------------------
# Google Quantum AI — Willow / making QEC work
run uv run chatqec ingest --url "https://research.google/blog/making-quantum-error-correction-work/"
# Google — developer's guide to error correction (surface code)
run uv run chatqec ingest --url "https://developers.googleblog.com/unlocking-the-potential-of-quantum-computing-a-developers-guide-to-error-correction/"
# IBM Quantum — fault-tolerant roadmap / qLDPC (Quantum Data Center 2025)
run uv run chatqec ingest --url "https://www.ibm.com/quantum/blog/qdc-2025"
# Quantinuum — path toward fault tolerance
run uv run chatqec ingest --url "https://www.quantinuum.com/blog/quantinuum-researchers-are-unlocking-a-more-efficient-and-powerful-path-towards-fault-tolerance"
# Quantinuum + CU Boulder — making QEC easier
run uv run chatqec ingest --url "https://www.quantinuum.com/blog/quantinuum-and-cu-boulder-just-made-quantum-error-correction-easier"

echo
echo "=== v0.4 seed complete ==="
echo "inspect registry:  uv run chatqec sources list"
echo "corpus stats:      uv run chatqec corpus-stats"
echo "embed phase:       uv run chatqec embed --all-pending --provider gemini"
