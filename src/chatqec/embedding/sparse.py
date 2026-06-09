"""BM25 sparse encoder via fastembed."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from fastembed import SparseTextEmbedding


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


class BM25Encoder:
    """Encodes text into BM25 sparse vectors usable directly by Qdrant."""

    DEFAULT_MODEL = "Qdrant/bm25"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model = SparseTextEmbedding(model_name=model_name)

    def encode_batch(self, texts: Sequence[str]) -> list[SparseVector]:
        embeddings = list(self._model.embed(list(texts)))
        out: list[SparseVector] = []
        for e in embeddings:
            out.append(
                SparseVector(
                    indices=list(map(int, e.indices)),
                    values=list(map(float, e.values)),
                )
            )
        return out
