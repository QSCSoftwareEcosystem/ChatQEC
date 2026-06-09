# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project scaffolding (uv, Hydra, Click CLI shell, pytest, ruff, mypy)
- ChatQEC public API stub returning placeholder Answer objects
- GitLab CI with lint + test stages
- Pre-commit hooks
- Design spec and implementation plan
- Source registry (`sources/registry.jsonl`) with typed `SourceRecord`, lifecycle states (`active|superseded|retracted|stale|pending_review`), and validated state transitions
- SHA-256 content-hash utilities (`chatqec.ingestion.hashing`)
- Canonical source-ID generators per source type (`chatqec.ingestion.ids`)
- CLI: `chatqec sources list/show/mark-retracted/compact`
- Canonical wiki layer: `CanonicalFrontmatter` schema, directory loader with cross-reference validation, section-based chunker
- 5 P0 canonical pages: surface-code, stabilizer-formalism, threshold-theorem, css-codes, magic-state-distillation
- 52-page canonical roster documentation (`knowledge/canonical/_roster.yaml`)
- CLI: `chatqec canonical list/validate`
- CI: validate-canonical stage runs on every push
- Ingestion pipeline:
  - `Chunk` model + on-disk `ChunkSink` writing `data/chunks/<source_id>.jsonl`
  - Math-aware chunker that protects `$...$`, `$$...$$`, and `\begin{X}...\end{X}` environments
  - arXiv title-and-author **resolver** (spec §7.4) — the architectural fix to the verification finding
  - Fetchers: arXiv (Atom + chunk), GitHub (files+repo metadata → chunks), YouTube (60s-window transcripts), URL/blog (trafilatura → markdown), manual PDF (unstructured)
  - Watchlist (`sources/watchlist.yaml`) schema + parser with seed entries for arxiv categories, GitHub repos, YouTube channels, and RSS feeds
- CLI:
  - `chatqec ingest --url / --resolve-by-title / --arxiv` (the `--arxiv <id>` path warns per spec §7.4)
  - `chatqec watch refresh --dry-run`
- Test infrastructure: `pytest -m network` marker; optional `network-tests` GitLab CI job (manual trigger)
- Embedding layer:
  - `Embedder` protocol with `GeminiEmbedder` (text-embedding-004) and `LocalEmbedder` (Nomic v1.5 via sentence-transformers)
  - BM25 sparse encoder via fastembed
- Qdrant store wrapper (`ChatQecStore`) with hybrid dense + sparse vectors, payload indexing on `status / source_id / authority_tier / topic_slugs`, status filtering, source-id deletion
- `EmbeddingPipeline` that reads `data/chunks/*.jsonl`, embeds, upserts to Qdrant, and promotes `pending_review` → `active`
- `docker-compose.yml` for local Qdrant
- CLI: `chatqec embed --source-id / --all-pending`
- M4 deferred paths wired up: `chatqec ingest --arxiv <id>` (warns per §7.4 and ingests), `chatqec sources compact --apply` (purges from Qdrant + registry)
- Retrieval pipeline:
  - Score normalization helpers (`min_max_normalize`, `identity_if_single`)
  - Spec §8.5 weighted-sum scoring (`α=0.5·dense + β=0.3·bm25 + γ=0.15·tier + δ=0.05·recency`) with tunable coefficients
  - Canonical topic matcher (`CanonicalIndex` + `match_topics`) that surfaces force-injection candidates by slug/alias substring
  - `CrossEncoderReranker` wrapping BAAI/bge-reranker-base with lazy model load
  - `HybridRetriever` orchestrator: dense + BM25 → score → top-K candidates → rerank → top-K results + matched topics
- CLI: `chatqec retrieve --query` debug command (Rich table of top-K + matched canonical topics)
- Conversation & synthesis layer (no tools yet — Stim lands in M8):
  - `ConversationMemory` for in-session message history (anonymous, in-memory)
  - `OffTopicClassifier` (Haiku) cheap pre-filter that refuses non-QEC queries
  - `QueryRewriter` (Haiku) for pronoun resolution + 0-3 retrieval variants
  - System prompt + context assembler (numbered `[^N]` footnote scheme, canonical chunks first)
  - `Synthesizer` (Sonnet) with `.synthesize()` and `.synthesize_stream()`
  - Footnote-marker citation extraction → `Citation` objects
  - Confidence heuristic: high (≥3 citations) / medium / low (no context)
- `ChatQEC` public class fully wired (DI constructor; `_build_chatqec` Hydra-config factory in CLI)
- CLI:
  - `chatqec ask "..."` runs the real pipeline and prints citations + confidence
  - `chatqec chat` interactive REPL with `/quit` and `/clear` commands
