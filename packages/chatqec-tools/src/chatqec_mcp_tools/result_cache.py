"""Opt-in on-disk cache for seeded tool calls (benchmark runs only).

A benchmark asks the same seeded question of several models. When two models extract the same
arguments, the tool result is identical by construction, so recomputing it (minutes of decoding
per call) only costs time. With ``CHATQEC_TOOL_CACHE_DIR`` set, ``cached_call`` stores each
successful result of a call that carries a non-null ``seed``, keyed by the tool name, the
canonical arguments and a hash of this package's source code, so any code change starts a
fresh cache. Unseeded calls, failed results and runs without the variable are never cached.
A hit is marked with ``"_cache_hit": true`` in the returned output.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

CACHE_ENV = "CHATQEC_TOOL_CACHE_DIR"
Handler = Callable[[dict[str, Any]], dict[str, Any]]


def source_hash(package_dir: Path) -> str:
    """sha256 over every .py file under ``package_dir`` (path and content), in sorted order."""
    digest = hashlib.sha256()
    for path in sorted(package_dir.rglob("*.py")):
        digest.update(str(path.relative_to(package_dir)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _key(name: str, args: dict[str, Any], code_hash: str) -> str:
    canonical = json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(f"{code_hash}\0{name}\0{canonical}".encode()).hexdigest()


def cached_call(
    name: str,
    args: dict[str, Any],
    handler: Handler,
    *,
    cache_dir: Path | None,
    code_hash: str,
) -> dict[str, Any]:
    if cache_dir is None or args.get("seed") is None:
        return handler(args)
    path = cache_dir / f"{_key(name, args, code_hash)}.json"
    if path.exists():
        try:
            hit = json.loads(path.read_text())
        except (OSError, ValueError):
            hit = None
        if isinstance(hit, dict):
            return {**hit, "_cache_hit": True}
    result = handler(args)
    if result.get("ok", True):
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Write then rename, so a concurrent reader never sees a half-written file.
        with tempfile.NamedTemporaryFile("w", dir=cache_dir, delete=False, suffix=".tmp") as f:
            json.dump(result, f)
        os.replace(f.name, path)
    return result


def cache_dir_from_env() -> Path | None:
    value = os.environ.get(CACHE_ENV)
    return Path(value) if value else None
