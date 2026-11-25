#!/usr/bin/env bash
# One-time Codespace setup: uv, Qdrant, the app, the CPU tools, and the corpus index.
# Also usable on any Linux x86_64 machine from the repository root.
set -euo pipefail

QDRANT_VERSION="1.12.0"   # same as docker-compose.yml
# CPU-only tool extras. Left out: cudaq (needs a GPU), mle (Gurobi licence),
# guppy (runs user programs; off by default anyway).
TOOL_EXTRAS="lightstim,tsim,ibm,decoders,paulice,ft,deltakit,decoderbench,ftprimitive"

cd "$(dirname "$0")/.."
mkdir -p "$HOME/.local/bin"
export PATH="$HOME/.local/bin:$PATH"

echo "==> uv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

echo "==> qdrant $QDRANT_VERSION"
if ! command -v qdrant >/dev/null; then
  curl -fsSL "https://github.com/qdrant/qdrant/releases/download/v${QDRANT_VERSION}/qdrant-x86_64-unknown-linux-gnu.tar.gz" \
    | tar -xz -C "$HOME/.local/bin" qdrant
fi

echo "==> ChatQEC app"
uv sync

echo "==> QEC tools (chatqec-tools)"
uv tool install --force --python 3.11 "./packages/chatqec-tools[${TOOL_EXTRAS}]"

echo "==> corpus index"
if [ -n "${CHATQEC_INDEX_SNAPSHOT:-}" ]; then
  uv run chatqec bootstrap --snapshot "$CHATQEC_INDEX_SNAPSHOT"
elif [ -n "${GEMINI_API_KEY:-}${GOOGLE_API_KEY:-}" ]; then
  uv run chatqec bootstrap
else
  echo "No GEMINI_API_KEY set, so the index was not built. Add it as a Codespaces"
  echo "secret (or export it) and run:  uv run chatqec bootstrap"
fi

echo "==> done. Start the app with:  bash .devcontainer/start.sh"
