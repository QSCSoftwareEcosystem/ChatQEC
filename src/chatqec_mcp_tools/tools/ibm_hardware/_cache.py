"""File-based TTL cache for IBM Quantum data. New pattern for this repo -- no
existing cache to model here, since every other tool in chatqec_mcp_tools is a
pure, stateless computation."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CacheResult:
    payload: Any
    fetched_at: float
    age_seconds: float
    stale: bool = False


def _cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    root = Path(base) / "chatqec-mcp-tools" / "ibm_hardware"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cache_path(key: str) -> Path:
    safe = key.replace("/", "__")
    return _cache_root() / f"{safe}.json"


def get_or_fetch(key: str, ttl_seconds: float, fetch_fn: Callable[[], Any]) -> CacheResult:
    """Return cached payload if fresh; otherwise call fetch_fn and persist the
    result. If fetch_fn raises and a (possibly expired) cached payload exists,
    serve it with stale=True instead of failing. If fetch_fn raises and there is
    no cached payload at all, the exception propagates."""
    path = _cache_path(key)
    now = time.time()
    cached_raw: dict[str, Any] | None = None

    if path.exists():
        try:
            cached_raw = json.loads(path.read_text(encoding="utf-8"))
            fetched_at = float(cached_raw["fetched_at"])
            age = now - fetched_at
            if age < ttl_seconds:
                return CacheResult(
                    payload=cached_raw["payload"], fetched_at=fetched_at, age_seconds=age
                )
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            cached_raw = None

    try:
        payload = fetch_fn()
    except Exception:
        if cached_raw is not None:
            fetched_at = float(cached_raw["fetched_at"])
            return CacheResult(
                payload=cached_raw["payload"],
                fetched_at=fetched_at,
                age_seconds=now - fetched_at,
                stale=True,
            )
        raise

    fetched_at = time.time()
    path.write_text(
        json.dumps({"fetched_at": fetched_at, "ttl_seconds": ttl_seconds, "payload": payload}),
        encoding="utf-8",
    )
    return CacheResult(payload=payload, fetched_at=fetched_at, age_seconds=0.0)
