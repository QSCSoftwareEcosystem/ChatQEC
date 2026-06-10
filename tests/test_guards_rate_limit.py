"""Per-IP rate limiter (SQLite-backed)."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from chatqec.guards.rate_limit import LimitExceeded, RateLimiter


def _now() -> datetime:
    return datetime(2026, 5, 29, 12, 0, tzinfo=UTC)


def test_rate_limiter_allows_under_cap(tmp_path: Path):
    rl = RateLimiter(
        db_path=tmp_path / "rl.db",
        limits={"ask": (20, "day"), "tool": (5, "hour")},
    )
    for _ in range(5):
        rl.check_and_record("1.2.3.4", action="ask", now=_now())


def test_rate_limiter_raises_when_daily_cap_hit(tmp_path: Path):
    rl = RateLimiter(db_path=tmp_path / "rl.db", limits={"ask": (3, "day")})
    for _ in range(3):
        rl.check_and_record("1.2.3.4", action="ask", now=_now())
    with pytest.raises(LimitExceeded):
        rl.check_and_record("1.2.3.4", action="ask", now=_now())


def test_rate_limiter_resets_after_window(tmp_path: Path):
    rl = RateLimiter(db_path=tmp_path / "rl.db", limits={"ask": (1, "hour")})
    rl.check_and_record("1.2.3.4", action="ask", now=_now())
    with pytest.raises(LimitExceeded):
        rl.check_and_record("1.2.3.4", action="ask", now=_now())
    later = _now() + timedelta(hours=1, minutes=1)
    rl.check_and_record("1.2.3.4", action="ask", now=later)


def test_rate_limiter_independent_per_ip(tmp_path: Path):
    rl = RateLimiter(db_path=tmp_path / "rl.db", limits={"ask": (1, "day")})
    rl.check_and_record("1.1.1.1", action="ask", now=_now())
    rl.check_and_record("2.2.2.2", action="ask", now=_now())


def test_rate_limiter_independent_per_action(tmp_path: Path):
    rl = RateLimiter(
        db_path=tmp_path / "rl.db",
        limits={"ask": (1, "day"), "tool": (1, "hour")},
    )
    rl.check_and_record("1.2.3.4", action="ask", now=_now())
    rl.check_and_record("1.2.3.4", action="tool", now=_now())


def test_remaining_returns_balance_for_each_action(tmp_path: Path):
    rl = RateLimiter(
        db_path=tmp_path / "rl.db",
        limits={"ask": (10, "day"), "tool": (2, "hour")},
    )
    rl.check_and_record("1.2.3.4", action="ask", now=_now())
    rl.check_and_record("1.2.3.4", action="ask", now=_now())
    assert rl.remaining("1.2.3.4", action="ask", now=_now()) == 8
    assert rl.remaining("1.2.3.4", action="tool", now=_now()) == 2
