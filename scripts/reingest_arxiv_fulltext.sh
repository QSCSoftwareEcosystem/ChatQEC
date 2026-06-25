#!/usr/bin/env bash
# Re-ingest the 13 foundational arXiv papers as FULL TEXT (LaTeX source, or
# Marker PDF fallback) and re-embed them, replacing the abstract-only records.
#
# Surgical: only the arXiv sources are touched. The 5 errorcorrectionzoo blog
# sources and their vectors are left intact.
#
# Prereqs:
#   - Qdrant running (scripts/qdrant_local.sh) with the existing chatqec_chunks
#   - GEMINI_API_KEY in .env
#
# Usage:
#   bash scripts/reingest_arxiv_fulltext.sh

set -euo pipefail
cd "$(dirname "$0")/.."

REGISTRY="sources/registry.jsonl"
COLLECTION="chatqec_chunks"
QDRANT_URL="http://localhost:6333"

ARXIV_IDS=(
  quant-ph/9707021 quant-ph/9601029 quant-ph/9906129 quant-ph/0403025
  quant-ph/0008040 1208.0928 1808.02892 2408.13687 2308.07915
  2409.17595 2107.02194 2201.03540 2101.09310
)

echo ">>> backing up registry to ${REGISTRY}.bak"
cp "$REGISTRY" "${REGISTRY}.bak"

echo ">>> pruning old arXiv records from registry, chunks, and Qdrant"
uv run python - "$REGISTRY" "$COLLECTION" "$QDRANT_URL" <<'PY'
import json, sys
from pathlib import Path
from qdrant_client import QdrantClient, models

registry_path, collection, qdrant_url = sys.argv[1], sys.argv[2], sys.argv[3]

# 1. Rewrite the registry keeping only non-arXiv records.
recs = [json.loads(l) for l in Path(registry_path).read_text().splitlines() if l.strip()]
arxiv_ids = [r["id"] for r in recs if r.get("type") == "arxiv"]
kept = [r for r in recs if r.get("type") != "arxiv"]
Path(registry_path).write_text("".join(json.dumps(r) + "\n" for r in kept))
print(f"  registry: dropped {len(arxiv_ids)} arXiv records, kept {len(kept)}")

# 2. Delete arXiv vectors from Qdrant.
client = QdrantClient(url=qdrant_url)
for sid in arxiv_ids:
    client.delete(
        collection_name=collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="source_id", match=models.MatchValue(value=sid))]
            )
        ),
    )
print(f"  qdrant: deleted vectors for {len(arxiv_ids)} arXiv sources")
PY

echo ">>> deleting old arXiv chunk files"
rm -f data/chunks/arxiv_*.jsonl

echo ">>> re-ingesting ${#ARXIV_IDS[@]} papers as full text"
for id in "${ARXIV_IDS[@]}"; do
  echo "  --- $id ---"
  uv run chatqec ingest --arxiv "$id" --full-text 2>&1 | grep -E "chunks|Marker|LaTeX|fallback|abstract" || true
done

echo ">>> embedding all pending_review sources"
uv run chatqec embed --all-pending --provider gemini

echo ">>> done. final source list:"
uv run chatqec sources list
