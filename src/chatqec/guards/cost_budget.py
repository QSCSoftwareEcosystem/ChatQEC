"""Daily USD-equivalent cost budget guard."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path


class BudgetExceeded(Exception):
    pass


_PRICES_USD_PER_MTOK = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cached_input": 0.30},
    "claude-haiku-4-5": {"input": 0.80, "output": 4.0, "cached_input": 0.08},
    "text-embedding-004": {"input": 0.025, "output": 0.0, "cached_input": 0.025},
}


def estimate_cost_usd(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float:
    p = _PRICES_USD_PER_MTOK.get(model, {"input": 1.0, "output": 1.0, "cached_input": 1.0})
    uncached_in = max(0, input_tokens - cached_input_tokens)
    return (
        uncached_in * p["input"] / 1_000_000
        + cached_input_tokens * p["cached_input"] / 1_000_000
        + output_tokens * p["output"] / 1_000_000
    )


class CostBudget:
    def __init__(self, *, db_path: Path, daily_usd_cap: float) -> None:
        self.db_path = Path(db_path)
        self.daily_usd_cap = daily_usd_cap
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _ensure_schema(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS spend (
                    day TEXT PRIMARY KEY, usd REAL NOT NULL DEFAULT 0.0
                )"""
            )

    def _key(self, now: datetime) -> str:
        return now.strftime("%Y-%m-%d")

    def spent_today(self, *, now: datetime) -> float:
        with self._conn() as conn:
            row = conn.execute("SELECT usd FROM spend WHERE day=?", (self._key(now),)).fetchone()
            return float(row[0]) if row else 0.0

    def remaining(self, *, now: datetime) -> float:
        return max(0.0, self.daily_usd_cap - self.spent_today(now=now))

    def charge(self, *, amount_usd: float, now: datetime) -> None:
        spent = self.spent_today(now=now)
        if spent + amount_usd > self.daily_usd_cap:
            raise BudgetExceeded(
                f"daily cap ${self.daily_usd_cap:.2f} would be exceeded "
                f"(spent ${spent:.3f} + charge ${amount_usd:.3f})"
            )
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO spend(day, usd) VALUES (?, ?)
                   ON CONFLICT(day) DO UPDATE SET usd = usd + excluded.usd""",
                (self._key(now), amount_usd),
            )
