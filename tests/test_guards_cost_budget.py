"""Daily token-budget guard tests."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from chatqec.guards.cost_budget import BudgetExceeded, CostBudget


def _now() -> datetime:
    return datetime(2026, 5, 29, 12, 0, tzinfo=UTC)


def test_budget_records_costs_and_allows_under_cap(tmp_path: Path):
    b = CostBudget(db_path=tmp_path / "b.db", daily_usd_cap=1.0)
    b.charge(amount_usd=0.05, now=_now())
    assert b.spent_today(now=_now()) == pytest.approx(0.05)


def test_budget_raises_when_cap_hit(tmp_path: Path):
    b = CostBudget(db_path=tmp_path / "b.db", daily_usd_cap=0.10)
    b.charge(amount_usd=0.08, now=_now())
    with pytest.raises(BudgetExceeded):
        b.charge(amount_usd=0.05, now=_now())


def test_budget_resets_at_utc_midnight(tmp_path: Path):
    b = CostBudget(db_path=tmp_path / "b.db", daily_usd_cap=0.10)
    b.charge(amount_usd=0.08, now=_now())
    tomorrow = _now() + timedelta(days=1)
    b.charge(amount_usd=0.08, now=tomorrow)
    assert b.spent_today(now=tomorrow) == pytest.approx(0.08)


def test_budget_remaining_returns_difference(tmp_path: Path):
    b = CostBudget(db_path=tmp_path / "b.db", daily_usd_cap=2.00)
    b.charge(amount_usd=0.50, now=_now())
    assert b.remaining(now=_now()) == pytest.approx(1.50)


def test_estimate_cost_from_tokens_matches_spec():
    from chatqec.guards.cost_budget import estimate_cost_usd

    cost = estimate_cost_usd(
        model="claude-sonnet-4-6",
        input_tokens=8000,
        output_tokens=800,
        cached_input_tokens=6000,
    )
    assert 0.005 <= cost <= 0.040
