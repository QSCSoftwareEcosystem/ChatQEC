"""SQLite-backed per-IP per-action rate limiter.

Windows: "day" (UTC midnight reset) or "hour" (rolling hour bucket).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Literal

Window = Literal["day", "hour"]


class LimitExceeded(Exception):
    pass


def _bucket(now: datetime, window: Window) -> str:
    if window == "day":
        return now.strftime("%Y-%m-%d")
    return now.strftime("%Y-%m-%dT%H")


class RateLimiter:
    def __init__(
        self,
        *,
        db_path: Path,
        limits: dict[str, tuple[int, Window]],
    ) -> None:
        self.db_path = Path(db_path)
        self.limits = limits
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _ensure_schema(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS counters (
                    ip TEXT NOT NULL,
                    action TEXT NOT NULL,
                    bucket TEXT NOT NULL,
                    count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (ip, action, bucket)
                )"""
            )

    def _current_count(self, ip: str, action: str, bucket: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT count FROM counters WHERE ip=? AND action=? AND bucket=?",
                (ip, action, bucket),
            ).fetchone()
            return int(row[0]) if row else 0

    def remaining(self, ip: str, *, action: str, now: datetime) -> int:
        cap, window = self.limits[action]
        bucket = _bucket(now, window)
        return max(0, cap - self._current_count(ip, action, bucket))

    def check_and_record(self, ip: str, *, action: str, now: datetime) -> None:
        cap, window = self.limits[action]
        bucket = _bucket(now, window)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT count FROM counters WHERE ip=? AND action=? AND bucket=?",
                (ip, action, bucket),
            ).fetchone()
            count = int(row[0]) if row else 0
            if count >= cap:
                raise LimitExceeded(f"limit {cap}/{window} exceeded for ip={ip} action={action}")
            if row:
                conn.execute(
                    "UPDATE counters SET count=count+1 WHERE ip=? AND action=? AND bucket=?",
                    (ip, action, bucket),
                )
            else:
                conn.execute(
                    "INSERT INTO counters(ip, action, bucket, count) VALUES (?, ?, ?, 1)",
                    (ip, action, bucket),
                )
