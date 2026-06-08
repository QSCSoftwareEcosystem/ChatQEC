"""SHA-256 helpers for the source registry's content-hash refresh logic."""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 65536  # 64 KiB streaming read


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(_CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()
