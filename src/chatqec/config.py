"""Frozen dataclass mirrors of the Hydra config; safer for downstream code."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    version: str


@dataclass(frozen=True)
class ModelsConfig:
    rewriter: str
    classifier: str
    synthesizer: str
    embedder: str
    long_context_fallback: str
    reranker: str


@dataclass(frozen=True)
class RetrieverConfig:
    top_k: int
    candidate_k: int
    alpha_vector: float
    beta_bm25: float
    gamma_tier: float
    delta_recency: float


@dataclass(frozen=True)
class ChatQECConfig:
    project: ProjectConfig
    models: ModelsConfig
    retriever: RetrieverConfig
