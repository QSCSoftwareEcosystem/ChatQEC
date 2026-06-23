#!/usr/bin/env bash
# Run Qdrant as a local process (no Docker, no sudo).
# Storage persists in qdrant_data/ at the repo root (already in .gitignore).
#
# Usage:
#   bash scripts/qdrant_local.sh        # run in foreground (Ctrl-C to stop)
#   bash scripts/qdrant_local.sh &      # background
#   curl http://localhost:6333/healthz  # verify

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p qdrant_data

# QDRANT__... env vars override the embedded default config.
QDRANT__SERVICE__HTTP_PORT=6333 \
QDRANT__SERVICE__GRPC_PORT=6334 \
QDRANT__STORAGE__STORAGE_PATH="$(pwd)/qdrant_data/storage" \
QDRANT__STORAGE__SNAPSHOTS_PATH="$(pwd)/qdrant_data/snapshots" \
QDRANT__LOG_LEVEL=info \
exec "$HOME/.local/bin/qdrant"
