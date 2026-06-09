"""Cross-encoder reranker wrapper (BAAI/bge-reranker-base by default).

The model is loaded lazily — the first call to `.rerank()` triggers the
~400MB download. In tests we mock the underlying CrossEncoder class so the
download never happens.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Returns the indices of the top_k most relevant candidates given a query."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    def _model_or_load(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: list[str], *, top_k: int) -> list[int]:
        if not candidates:
            return []
        model = self._model_or_load()
        pairs = [[query, c] for c in candidates]
        scores: np.ndarray = model.predict(pairs)
        top_k = min(top_k, len(candidates))
        # Indices of top_k highest scores, sorted descending
        order = np.argsort(-scores)[:top_k]
        return [int(i) for i in order]
