"""Call the IQM worker (``_worker.py``) in its own interpreter.

iqm-client cannot be installed next to the other extras (it pins packaging<25 and
pydantic<2.13; deltakit needs packaging>=25), so the IQM SDK lives in a separate venv made
by ``scripts/setup_iqm_env.sh``. This module only spawns that interpreter and parses its
JSON reply; it never imports iqm-client.

The child gets a minimal environment: the IQM_* settings plus HOME/PATH/LANG, so no other
API key from the server's environment is passed on.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

_WORKER = Path(__file__).with_name("_worker.py")
_DEFAULT_PYTHON = Path.home() / ".local/share/chatqec-mcp-tools/iqm-env/bin/python"
_PASSTHROUGH = ("IQM_TOKEN", "IQM_SERVER_URL", "IQM_QUANTUM_COMPUTER", "HOME", "PATH", "LANG")
_DEFAULT_TIMEOUT_SECONDS = 120.0

IQM_SETUP_HINT = (
    "Create the IQM environment once with `bash ~/chatqec-mcp-tools/scripts/setup_iqm_env.sh` "
    "(or point CHATQEC_IQM_PYTHON at a Python that has iqm-client[qiskit])."
)


class IQMCredentialsError(RuntimeError):
    """Missing/invalid IQM credentials or a missing IQM environment. Not transient, so the
    cache never serves a stale snapshot in its place."""


class IQMWorkerError(RuntimeError):
    """The worker ran but the IQM call failed (network, server, bad circuit, ...)."""


def iqm_python() -> Path:
    return Path(os.environ.get("CHATQEC_IQM_PYTHON") or _DEFAULT_PYTHON)


def call_worker(op: str, timeout: float | None = None, **fields: Any) -> dict[str, Any]:
    """Run one worker op and return its ``result`` dict.

    Raises IQMCredentialsError (no token/URL, auth refused, no IQM env) or IQMWorkerError.
    """
    if not os.environ.get("IQM_TOKEN"):
        raise IQMCredentialsError("IQM_TOKEN is not set in the environment (.env).")
    if not os.environ.get("IQM_SERVER_URL"):
        raise IQMCredentialsError(
            "IQM_SERVER_URL is not set (e.g. IQM_SERVER_URL=https://qccsw.ccs.ornl.gov/ in .env)."
        )
    python = iqm_python()
    if not python.exists():
        raise IQMCredentialsError(f"IQM Python not found at {python}. {IQM_SETUP_HINT}")

    env = {k: os.environ[k] for k in _PASSTHROUGH if k in os.environ}
    env["PYTHONUNBUFFERED"] = "1"
    wall = timeout or float(os.environ.get("CHATQEC_IQM_TIMEOUT_SECONDS", _DEFAULT_TIMEOUT_SECONDS))
    try:
        proc = subprocess.run(
            [str(python), "-I", str(_WORKER)],
            input=json.dumps({"op": op, **fields}),
            capture_output=True,
            text=True,
            env=env,
            timeout=wall,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise IQMWorkerError(f"IQM call '{op}' timed out after {wall:.0f} s.") from exc

    try:
        reply = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        tail = proc.stderr.strip().splitlines()[-3:]
        raise IQMWorkerError(
            f"IQM worker gave no reply (exit {proc.returncode}): {' | '.join(tail)}"
        ) from exc

    if reply.get("ok"):
        return dict(reply["result"])
    if reply.get("kind") == "auth":
        raise IQMCredentialsError(f"IQM authentication failed: {reply.get('error', '')}")
    raise IQMWorkerError(str(reply.get("error", "unknown IQM error")))


__all__ = [
    "IQM_SETUP_HINT",
    "IQMCredentialsError",
    "IQMWorkerError",
    "call_worker",
    "iqm_python",
]
