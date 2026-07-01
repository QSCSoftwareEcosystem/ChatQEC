"""Corpus statistics over the ingested chunk store.

Descriptive stats for reporting / papers: chunk counts, token-length
distribution (total / min / max / mean / median / p95), per-source breakdown,
and splits by source type and authority tier.

Token counts are **approximate** — computed with the same ``chars / 4`` unit the
math-aware chunker targets (so a "512-token" chunk here matches the chunker's
own budget). This keeps the stats dependency-free and reproducible; swap in a
real BPE tokenizer later if exact model-token counts are needed.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

_CHARS_PER_TOKEN = 4  # matches chatqec.ingestion.chunkers.math_aware


def approx_tokens(text: str) -> int:
    """Approximate token count (ceil of chars / 4)."""
    return (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN


def _summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "total": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0.0, "p95": 0}
    ordered = sorted(values)
    p95_idx = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return {
        "n": len(values),
        "total": sum(values),
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 1),
        "median": round(statistics.median(values), 1),
        "p95": ordered[p95_idx],
    }


def compute_corpus_stats(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute corpus-wide + per-source statistics from chunk records."""
    tokens = [approx_tokens(str(c.get("text", ""))) for c in chunks]
    chars = [len(str(c.get("text", ""))) for c in chunks]

    by_source: dict[str, list[int]] = {}
    for c, tok in zip(chunks, tokens, strict=True):
        by_source.setdefault(str(c.get("source_id", "?")), []).append(tok)

    largest = None
    if chunks:
        hi = max(range(len(chunks)), key=lambda i: tokens[i])
        largest = {
            "chunk_id": chunks[hi].get("chunk_id"),
            "source_id": chunks[hi].get("source_id"),
            "title": chunks[hi].get("title"),
            "tokens": tokens[hi],
            "chars": chars[hi],
        }

    return {
        "n_sources": len(by_source),
        "n_chunks": len(chunks),
        "tokens": _summary(tokens),
        "chars": _summary(chars),
        "chunks_per_source": _summary([len(v) for v in by_source.values()]),
        "by_source_type": dict(Counter(str(c.get("source_type", "?")) for c in chunks)),
        "by_tier": dict(Counter(str(c.get("authority_tier", "?")) for c in chunks)),
        "largest_chunk": largest,
        "per_source": {sid: _summary(v) for sid, v in sorted(by_source.items())},
        "token_unit": f"approx (chars/{_CHARS_PER_TOKEN})",
    }


def load_chunks(chunks_dir: Path) -> list[dict[str, Any]]:
    """Load all chunk records from ``<chunks_dir>/*.jsonl``."""
    out: list[dict[str, Any]] = []
    for path in sorted(chunks_dir.glob("*.jsonl")):
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line:
                out.append(json.loads(line))
    return out
