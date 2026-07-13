"""Spec §8.5 weighted-sum candidate scoring tests."""

from datetime import datetime, UTC

import pytest

from chatqec.retrieval.scoring import (
    Candidate,
    Coefficients,
    TIER_WEIGHTS,
    compute_recency,
    score_candidates,
)


def _now() -> datetime:
    return datetime(2026, 5, 29, tzinfo=UTC)


def test_tier_weights_match_spec():
    # spec §8.5
    assert TIER_WEIGHTS["canonical"] == 1.0
    assert TIER_WEIGHTS["peer_reviewed"] == 0.9
    assert TIER_WEIGHTS["preprint"] == 0.7
    assert TIER_WEIGHTS["blog"] == 0.5
    assert TIER_WEIGHTS["video"] == 0.5


def test_compute_recency_returns_one_for_current_year():
    assert compute_recency(year=2026, now=_now()) == 1.0


def test_compute_recency_drops_to_zero_after_five_years():
    assert compute_recency(year=2020, now=_now()) == 0.0
    assert compute_recency(year=1999, now=_now()) == 0.0


def test_compute_recency_handles_missing_year():
    assert compute_recency(year=None, now=_now()) == 0.5  # neutral


def test_score_uses_default_coefficients_from_spec():
    cands = [
        Candidate(
            chunk_id="c1", dense_norm=1.0, bm25_norm=0.0, authority_tier="canonical", year=2026
        ),
        Candidate(chunk_id="c2", dense_norm=0.0, bm25_norm=1.0, authority_tier="blog", year=2020),
    ]
    scored = score_candidates(cands, now=_now())
    # c1: 0.5*1 + 0.3*0 + 0.15*1.0 + 0.05*1 = 0.70
    # c2: 0.5*0 + 0.3*1 + 0.15*0.5 + 0.05*0  = 0.375
    assert scored[0].chunk_id == "c1"
    assert scored[0].final_score == pytest.approx(0.70)
    assert scored[1].chunk_id == "c2"
    assert scored[1].final_score == pytest.approx(0.375)


def test_score_with_custom_coefficients():
    coeffs = Coefficients(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0)
    cands = [
        Candidate(chunk_id="c1", dense_norm=0.7, bm25_norm=0.0, authority_tier="blog", year=2024),
    ]
    [s] = score_candidates(cands, coefficients=coeffs, now=_now())
    assert s.final_score == pytest.approx(0.7)


def test_score_sorts_results_desc():
    cands = [
        Candidate(
            chunk_id="c_low", dense_norm=0.1, bm25_norm=0.0, authority_tier="blog", year=2020
        ),
        Candidate(
            chunk_id="c_high", dense_norm=0.9, bm25_norm=0.9, authority_tier="canonical", year=2026
        ),
    ]
    scored = score_candidates(cands, now=_now())
    assert scored[0].chunk_id == "c_high"
    assert scored[-1].chunk_id == "c_low"


def test_web_tier_ranks_below_blog():
    assert TIER_WEIGHTS["web"] < TIER_WEIGHTS["blog"]


def test_unknown_tier_falls_back_to_blog_weight():
    cands = [
        Candidate(
            chunk_id="x", dense_norm=0.5, bm25_norm=0.5, authority_tier="unknown_tier", year=2025
        ),
    ]
    [s] = score_candidates(cands, now=_now())
    # tier_weight = 0.5 (blog fallback)
    expected = 0.5 * 0.5 + 0.3 * 0.5 + 0.15 * 0.5 + 0.05 * (4 / 5)
    assert s.final_score == pytest.approx(expected)
