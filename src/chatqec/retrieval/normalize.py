"""Score normalization helpers.

Dense (cosine) and BM25 scores live on different scales; bring both to [0, 1]
before applying the weighted-sum scoring in §8.5.
"""

from __future__ import annotations


def min_max_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    span = hi - lo
    return [(s - lo) / span for s in scores]


def identity_if_single(scores: list[float]) -> list[float]:
    """Like min_max but treats single-element input as relevance=1.0."""
    if len(scores) == 1:
        return [1.0]
    return min_max_normalize(scores)
