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
# -----------------------------------------------------------------------------
run uv run chatqec ingest --resolve-by-title "Good quantum error-correcting codes exist" \
  --author "Calderbank Shor" --year 1996 --full-text --describe-figures
run uv run chatqec ingest --resolve-by-title "A Theory of Quantum Error-Correcting Codes" \
  --author "Knill Laflamme" --year 1997 --full-text --describe-figures
run uv run chatqec ingest --resolve-by-title "Topological quantum memory" \
  --author "Dennis Kitaev Landahl Preskill" --year 2002 --full-text --describe-figures
run uv run chatqec ingest --resolve-by-title "Topological quantum distillation" \
  --author "Bombin Martin-Delgado" --year 2006 --full-text --describe-figures
run uv run chatqec ingest --resolve-by-title "Magic state distillation with low overhead" \
  --author "Bravyi Haah" --year 2012 --full-text --describe-figures
run uv run chatqec ingest --resolve-by-title "Magic State Distillation Not as Costly as You Think" \
  --author "Litinski" --year 2019 --full-text --describe-figures
run uv run chatqec ingest --resolve-by-title "Fault-tolerant quantum computation with constant overhead" \
  --author "Gottesman" --year 2014 --full-text --describe-figures
run uv run chatqec ingest --resolve-by-title "Asymptotically good quantum and locally testable classical LDPC codes" \
  --author "Panteleev Kalachev" --year 2022 --full-text --describe-figures

# -----------------------------------------------------------------------------
# Landmark hardware fault-tolerance demonstrations (full LaTeX + figures)
# -----------------------------------------------------------------------------
run uv run chatqec ingest --resolve-by-title "Suppressing quantum errors by scaling a surface code logical qubit" \
  --author "Google Quantum AI" --year 2023 --full-text --describe-figures
run uv run chatqec ingest --resolve-by-title "Realization of real-time fault-tolerant quantum error correction" \
  --author "Ryan-Anderson" --year 2021 --full-text --describe-figures
run uv run chatqec ingest --resolve-by-title "Logical quantum processor based on reconfigurable atom arrays" \
  --author "Bluvstein Lukin" --year 2023 --full-text --describe-figures

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
