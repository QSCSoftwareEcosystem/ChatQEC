"""Spec §8.5 candidate scoring.

Computes final_score = alpha*dense_norm + beta*bm25_norm + gamma*tier_weight + delta*recency
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

TIER_WEIGHTS: dict[str, float] = {
    "canonical": 1.0,
    "peer_reviewed": 0.9,
    "preprint": 0.7,
    "blog": 0.5,
    "video": 0.5,
}

_RECENCY_WINDOW_YEARS = 5


@dataclass(frozen=True)
class Coefficients:
    alpha: float = 0.5
    beta: float = 0.3
    gamma: float = 0.15
    delta: float = 0.05


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    dense_norm: float
    bm25_norm: float
    authority_tier: str
    year: int | None


@dataclass(frozen=True)
class ScoredCandidate:
    chunk_id: str
    final_score: float


def compute_recency(*, year: int | None, now: datetime) -> float:
    if year is None:
        return 0.5
    age = now.year - year
    if age <= 0:
        return 1.0
    if age >= _RECENCY_WINDOW_YEARS:
        return 0.0
    return 1.0 - (age / _RECENCY_WINDOW_YEARS)


def score_candidates(
    candidates: list[Candidate],
    *,
    coefficients: Coefficients | None = None,
    now: datetime | None = None,
) -> list[ScoredCandidate]:
    coefficients = coefficients or Coefficients()
    now = now or datetime.now(tz=UTC)
    out: list[ScoredCandidate] = []
    for c in candidates:
        tier_weight = TIER_WEIGHTS.get(c.authority_tier, TIER_WEIGHTS["blog"])
        recency = compute_recency(year=c.year, now=now)
        final = (
            coefficients.alpha * c.dense_norm
            + coefficients.beta * c.bm25_norm
            + coefficients.gamma * tier_weight
            + coefficients.delta * recency
        )
        out.append(ScoredCandidate(chunk_id=c.chunk_id, final_score=final))
    out.sort(key=lambda s: s.final_score, reverse=True)
    return out
