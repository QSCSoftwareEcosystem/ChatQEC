"""Frozen dataclass config wrappers around the Hydra config."""

import dataclasses
import pytest

from chatqec.config import RetrieverConfig, ModelsConfig, ProjectConfig, ChatQECConfig


def test_retriever_config_is_frozen():
    cfg = RetrieverConfig(
        top_k=8,
        candidate_k=50,
        alpha_vector=0.5,
        beta_bm25=0.3,
        gamma_tier=0.15,
        delta_recency=0.05,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.top_k = 16  # type: ignore[misc]


def test_models_config_holds_model_ids():
    cfg = ModelsConfig(
        rewriter="claude-haiku-4-5",
        classifier="claude-haiku-4-5",
        synthesizer="claude-sonnet-4-6",
        embedder="text-embedding-004",
        long_context_fallback="gemini-2.5-pro",
        reranker="BAAI/bge-reranker-base",
    )
    assert cfg.synthesizer.startswith("claude-")
    assert cfg.embedder == "text-embedding-004"


def test_chatqec_config_aggregates_subconfigs():
    cfg = ChatQECConfig(
        project=ProjectConfig(name="chatqec", version="0.1.0-dev"),
        models=ModelsConfig(
            rewriter="claude-haiku-4-5",
            classifier="claude-haiku-4-5",
            synthesizer="claude-sonnet-4-6",
            embedder="text-embedding-004",
            long_context_fallback="gemini-2.5-pro",
            reranker="BAAI/bge-reranker-base",
        ),
        retriever=RetrieverConfig(
            top_k=8,
            candidate_k=50,
            alpha_vector=0.5,
            beta_bm25=0.3,
            gamma_tier=0.15,
            delta_recency=0.05,
        ),
    )
    assert cfg.project.name == "chatqec"
