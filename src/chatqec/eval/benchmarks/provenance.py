"""Reproducibility record written as the first line of every benchmark run file.

Everything a reader needs to rerun the same evaluation, or to see why two runs differ: both
repositories' commits and dirty flags, the pinned benchmark commits, the gold file's hash, the
model the server actually served, the tool server's package versions, Python and GPU. Missing
pieces are recorded as "unavailable", never guessed.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

UNAVAILABLE = "unavailable"
_KEY_PACKAGES = (
    "stim", "pymatching", "ldpc", "mwpf", "relay-bp", "ft-primitive-bench", "qiskit",
    "qiskit-paulice", "deltakit", "guppylang", "chatqec-mcp-tools", "numpy",
)  # fmt: skip


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        done = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return UNAVAILABLE
    return done.stdout.strip()


def git_state(repo: Path) -> dict[str, Any]:
    head = _run(["git", "rev-parse", "HEAD"], cwd=repo)
    status = _run(["git", "status", "--porcelain"], cwd=repo)
    return {"path": str(repo), "commit": head, "dirty": bool(status) and status != UNAVAILABLE}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else UNAVAILABLE


def tool_env_versions(executable: str = "chatqec-mcp-tools") -> dict[str, str]:
    """Package versions inside the installed tool server's own environment."""
    exe = shutil.which(executable)
    if exe is None:
        return {"error": f"{executable} not on PATH"}
    first = Path(exe).read_text(errors="replace").splitlines()[0]
    python = first[2:].strip() if first.startswith("#!") else ""
    if not python:
        return {"error": "could not find the tool server's interpreter"}
    code = (
        "import importlib.metadata as m, json, sys\n"
        "out = {'python': sys.version.split()[0]}\n"
        f"for p in {list(_KEY_PACKAGES)!r}:\n"
        "    try: out[p] = m.version(p)\n"
        "    except m.PackageNotFoundError: out[p] = 'not installed'\n"
        "print(json.dumps(out))\n"
    )
    raw = _run([python, "-I", "-c", code])
    parsed = json.loads(raw) if raw.startswith("{") else None
    if not isinstance(parsed, dict):
        return {"error": raw}
    return {str(k): str(v) for k, v in parsed.items()}


def served_model(base_url: str) -> Any:
    """What an OpenAI-compatible server reports at /models (id, root path, max length)."""
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=5) as resp:
            return json.load(resp).get("data", UNAVAILABLE)
    except (OSError, ValueError):
        return UNAVAILABLE


def collect(
    *,
    chatqec_repo: Path,
    tools_repo: Path,
    pins: Path,
    gold: Path,
    model: str,
    model_url: str | None,
    settings: dict[str, Any],
) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "chatqec": git_state(chatqec_repo),
        "tools": git_state(tools_repo),
        "pins_sha256": sha256_file(pins),
        "pins": pins.read_text() if pins.exists() else UNAVAILABLE,
        "gold": {"path": str(gold), "sha256": sha256_file(gold)},
        "model": model,
        "served_model": served_model(model_url) if model_url else UNAVAILABLE,
        "tool_env": tool_env_versions(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpus": _run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "settings": settings,
    }
