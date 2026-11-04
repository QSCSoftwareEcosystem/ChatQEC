"""Cross-encoder reranker wrapper (mixedbread-ai/mxbai-rerank-base-v1 by default).

The default is an Apache-2.0 cross-encoder (Mixedbread AI, DeBERTa-v3 based)
chosen to avoid China-origin models such as ``BAAI/bge-reranker-*``. Note: only
the **v1** mxbai-rerank line is DeBERTa-based; mxbai-rerank-**v2** is built on a
Qwen base, so it is intentionally not used here.

The model is loaded lazily — the first call to ``.rerank()`` triggers the
~280MB download. In tests we mock the underlying CrossEncoder class so the
download never happens. The model name is configurable via ``models.reranker``.
"""

from __future__ import annotations

import logging

import numpy as np
from huggingface_hub.errors import HfHubHTTPError
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Returns the indices of the top_k most relevant candidates given a query."""

    def __init__(self, model_name: str = "mixedbread-ai/mxbai-rerank-base-v1") -> None:
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    def _model_or_load(self) -> CrossEncoder:
        if self._model is None:
            try:
                self._model = CrossEncoder(self.model_name)
            except HfHubHTTPError as hub_err:
                # A stale HF_TOKEN in the environment makes the hub answer 401
                # even for a public model; load_dotenv() never overrides it.
                # If the model is already cached, answering beats failing.
                try:
                    self._model = CrossEncoder(self.model_name, local_files_only=True)
                except OSError:
                    raise hub_err from None
                logger.warning(
                    "Hugging Face Hub refused %s (%s); loaded it from the local cache. "
                    "Check HF_TOKEN in your shell profile.",
                    self.model_name,
                    hub_err.__class__.__name__,
                )
        return self._model

    def rerank(self, query: str, candidates: list[str], *, top_k: int) -> list[int]:
        return [i for i, _ in self.rerank_scored(query, candidates, top_k=top_k)]

    def rerank_scored(
        self, query: str, candidates: list[str], *, top_k: int
    ) -> list[tuple[int, float]]:
        if not candidates:
            return []
        model = self._model_or_load()
        scores = model.predict([[query, c] for c in candidates])
        top_k = min(top_k, len(candidates))
        # Indices of top_k highest scores, sorted descending
        order = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in order]
