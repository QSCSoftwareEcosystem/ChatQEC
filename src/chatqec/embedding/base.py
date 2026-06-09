"""Embedder protocol and shared types."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol, runtime_checkable


class EmbeddingDim(IntEnum):
    GEMINI_004 = 768
    NOMIC_15 = 768


@dataclass(frozen=True)
class EmbedderConfig:
    provider: str  # "gemini" | "local"
    model: str
    batch_size: int
    local_model: str


@runtime_checkable
class Embedder(Protocol):
    """Anything with a `.dim` and `.embed_batch(texts)` is an embedder."""

    dim: int

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]: ...
