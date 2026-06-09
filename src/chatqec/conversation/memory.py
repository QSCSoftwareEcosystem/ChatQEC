"""In-session message history.

Per spec §9.1: anonymous sessions, in-memory only, no per-user accounts in v1.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Turn:
    role: Literal["user", "assistant"]
    content: str


class ConversationMemory:
    def __init__(self) -> None:
        self._turns: list[Turn] = []

    def add_user(self, content: str) -> None:
        self._turns.append(Turn(role="user", content=content))

    def add_assistant(self, content: str) -> None:
        self._turns.append(Turn(role="assistant", content=content))

    def turns(self) -> Iterable[Turn]:
        return list(self._turns)

    def recent(self, *, n: int) -> list[Turn]:
        return self._turns[-n:]

    def clear(self) -> None:
        self._turns = []
