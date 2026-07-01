#!/usr/bin/env bash
# Seed the ChatQEC v0.3 corpus: book chapters + open-access QEC reviews.
#
# Adds on top of the v0.2 corpus (arXiv papers + errorcorrectionzoo):
#   - Textbook chapters (via the new `chatqec ingest --pdf ... --pages` path,
#     extracted with Marker): Nielsen & Chuang Ch.10, Lidar & Brun (4 chapters),
#     Picturing Quantum Software Ch.12 (ZX view, incl. the surface code).
#   - Open-access, book-length reviews/theses via `--arxiv --full-text`.
#
# Phase 1 only — produces JSONL chunks + pending_review registry rows.
# Run Phase 2 (embed) separately after inspecting the output.
#
# NOTE: book PDFs live in Raw/ (personal copies; chapter excerpts only).
# Marker extraction is GPU-accelerated and can take a few minutes per chapter.
#
# Usage:
#   bash scripts/seed_corpus_v0_3.sh           # ingest everything
#   bash scripts/seed_corpus_v0_3.sh --dry     # echo commands only
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

LB="Raw/cambridge-core_quantum-error-correction_1Jul2026"
NC="Raw/quantum-computation-and-quantum-information-nielsen-chuang.pdf"
PQS="Raw/pqs.pdf"

# -----------------------------------------------------------------------------
# Textbook chapters (Marker PDF extraction, peer_reviewed tier)
# -----------------------------------------------------------------------------

# Nielsen & Chuang — Ch.10 "Quantum error-correction" (PDF pp. 459-533)
run uv run chatqec ingest --pdf "$NC" --pages 459-533 \
  --title "Nielsen & Chuang — Ch.10 Quantum Error-Correction" \
  --pdf-author "Nielsen, Chuang" --year 2010 --tier peer_reviewed \
  --source-license "Cambridge University Press (personal copy; chapter excerpt)"

# Lidar & Brun (eds.), Quantum Error Correction (Cambridge 2013) — per-chapter PDFs
run uv run chatqec ingest --pdf "$LB/07.2_pp_46_77_Introduction_to_quantum_error_correction.pdf" \
  --title "Lidar & Brun — Introduction to Quantum Error Correction" \
  --pdf-author "Gottesman" --year 2013 --tier peer_reviewed \
  --source-license "Cambridge University Press (personal copy; chapter)"

run uv run chatqec ingest --pdf "$LB/07.5_pp_126_160_Introduction_to_quantum_fault_tolerance.pdf" \
  --title "Lidar & Brun — Introduction to Quantum Fault Tolerance" \
  --pdf-author "Gottesman" --year 2013 --tier peer_reviewed \
  --source-license "Cambridge University Press (personal copy; chapter)"

run uv run chatqec ingest --pdf "$LB/08.1_pp_163_180_Operator_quantum_error_correction.pdf" \
  --title "Lidar & Brun — Operator Quantum Error Correction" \
  --pdf-author "Kribs, Spekkens" --year 2013 --tier peer_reviewed \
  --source-license "Cambridge University Press (personal copy; chapter)"

run uv run chatqec ingest --pdf "$LB/08.2_pp_181_200_Entanglement-assisted_quantum_error-correcting_codes.pdf" \
  --title "Lidar & Brun — Entanglement-Assisted Quantum Error-Correcting Codes" \
  --pdf-author "Wilde, Brun" --year 2013 --tier peer_reviewed \
  --source-license "Cambridge University Press (personal copy; chapter)"

# Picturing Quantum Software — Ch.12 "Fault-tolerant quantum computation" (ZX; PDF pp. 566-660)
run uv run chatqec ingest --pdf "$PQS" --pages 566-660 \
  --title "Picturing Quantum Software — Ch.12 Fault-Tolerant Quantum Computation (ZX)" \
  --pdf-author "Kissinger, van de Wetering" --year 2024 --tier peer_reviewed \
  --source-license "author draft (freely available online); chapter excerpt"

# -----------------------------------------------------------------------------
# Open-access, book-length reviews / theses (full LaTeX source via arXiv)
# -----------------------------------------------------------------------------
run uv run chatqec ingest --arxiv quant-ph/9705052 --full-text  # Gottesman thesis — Stabilizer Codes & QEC
run uv run chatqec ingest --arxiv 0905.2794 --full-text         # Devitt-Munro-Nemoto — QEC for Beginners
run uv run chatqec ingest --arxiv 1907.11157 --full-text        # Roffe — QEC: An Introductory Guide
run uv run chatqec ingest --arxiv 1302.3428 --full-text         # Terhal — QEC for quantum memories (RMP)

# --- Verify the arXiv id before enabling (2024-search additions) ---
# Girvin — Introduction to QEC and Fault Tolerance (Les Houches). Confirm id first.
# run uv run chatqec ingest --arxiv 2111.08894 --full-text
# Cleland — An Introduction to the Surface Code (SciPost, CC-BY). Not confirmed on arXiv;
# download the SciPost PDF into Raw/ and ingest via --pdf instead.

echo
echo "=== v0.3 seed complete ==="
echo "inspect registry:  uv run chatqec sources list"
echo "corpus stats:      uv run chatqec corpus-stats"
echo "embed phase:       see scripts/embed_corpus_v0_2.sh (or embed --all-pending)"
