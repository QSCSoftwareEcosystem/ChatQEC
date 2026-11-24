#!/usr/bin/env bash
# Create the separate Python environment the iqm_* tools run IQM calls in.
#
# iqm-client pins packaging<25 and pydantic<2.13, which conflicts with the deltakit and
# pydantic versions the main chatqec-mcp-tools install needs, so it cannot be an extra.
# The tools find this env at ~/.local/share/chatqec-mcp-tools/iqm-env (override with
# CHATQEC_IQM_PYTHON=/path/to/python).
set -euo pipefail

ENV_DIR="${1:-$HOME/.local/share/chatqec-mcp-tools/iqm-env}"
IQM_CLIENT_SPEC="${IQM_CLIENT_SPEC:-iqm-client[qiskit]>=35.0.3,<38}"

uv venv --python 3.12 "$ENV_DIR"
uv pip install --python "$ENV_DIR/bin/python" "$IQM_CLIENT_SPEC"
"$ENV_DIR/bin/python" -c "import importlib.metadata as m; print('iqm-client', m.version('iqm-client'), 'qiskit', m.version('qiskit'))"
echo "IQM env ready: $ENV_DIR/bin/python"
