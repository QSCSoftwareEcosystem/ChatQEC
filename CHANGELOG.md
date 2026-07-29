# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Pinned GitHub-repo ingestion: `chatqec ingest --github <org>/<repo> --commit-sha
  <sha> [--github-paths ...]` (`cli.py`), reusing the existing
  `GitHubMetadata`/`files_to_chunks` fetcher (`ingestion/fetchers/github.py`).
  `--commit-sha` is mandatory and validated as a 40-char hex commit — never a
  branch — before any network call; reads `GITHUB_PERSONAL_ACCESS_TOKEN`
  (this repo's established token env var, not `GITHUB_TOKEN`). Restricted to
  `.md` files (`filter_file_extensions`) so root-level repo plumbing
  (`Dockerfile`, `requirements.txt`, ...) doesn't ride along with the curated
  docs `--github-paths` targets. `QuTone/LightStim` added to
  `sources/watchlist.yaml` so `chatqec watch refresh` surfaces upstream moves.
  Live-run: `github:QuTone/LightStim@47dd7497` (79 chunks, `docs/api/` +
  `skills/`) and `arxiv:2604.21472` (39 chunks, full LaTeX source of the
  LightStim paper) both ingested and embedded — see
  [`docs/EXAMPLES.md`](docs/EXAMPLES.md#lightstim-knowledge-base-docs--paper-ingested-2026-07-28).
- `qec_circuit_build` MCP tool in the sibling `chatqec-mcp-tools` repo: builds
  circuits for code families Stim doesn't ship built-in (toric code, BB codes,
  PQRM, LightStim's construction of the 6.6.6 color code) by delegating to
  [LightStim](https://github.com/QuTone/LightStim), pinned to an exact commit
  via an optional `lightstim` extra (`chatqec-mcp-tools[lightstim]`) so the
  five existing Stim-backed tools keep working unmodified if it's absent.
  Also fixed an `mcp` dependency cap (`>=0.9,<2`) — an unconstrained upper
  bound let `uv tool install` resolve a breaking `mcp==2.0.0` whose `Server`
  class dropped the `list_tools`/`call_tool` API the real stdio server uses.
  Live capability check: 4 of a 20-prompt benchmark set
  (`eval/lightstim_qec_prompts.json`) run via real `chatqec ask` calls, all
  correct. See
  [`docs/EXAMPLES.md`](docs/EXAMPLES.md#lightstim-via-qec_circuit_build) and
  [`docs/lightstim_capability_check.md`](docs/lightstim_capability_check.md).
- `tsim_simulate` MCP tool in the sibling `chatqec-mcp-tools` repo: runs a
  Stim-superset circuit through [Tsim](https://github.com/QuEraComputing/tsim),
  a GPU-accelerated simulator that adds non-Clifford gates (`T`, `T_DAG`,
  `R_X`/`R_Y`/`R_Z`, `U3`, `TPP`/`TPP_DAG`, `R_XX`/`R_YY`/`R_ZZ`/`R_PAULI`,
  `CCZ`, `CCX`) via stabilizer rank decomposition — circuits plain `stim`
  itself rejects (`ValueError: Gate not found`). Depends on
  `bloqade-tsim>=0.1.5,<0.2` via an optional `tsim` extra
  (`chatqec-mcp-tools[tsim]`), a normal versioned PyPI dependency (no
  commit-pin needed, unlike LightStim's git-only dependency — see
  [`docs/DESIGN_CHOICES.md`](docs/DESIGN_CHOICES.md#why-tsim-needs-no-commit-pin-for-its-code-dependency-only-for-kb-ingestion)).
  Enforces a 256 KB circuit-size cap and a 100,000-shot cap (lower than
  `stim_simulate`'s 1,000,000, since Tsim's ZX/JAX pipeline is far more
  expensive per shot); supports `use_reference_sample` to switch between
  Tsim's raw detector convention and Stim's XOR-against-noiseless-reference
  convention.
- `glcb_visualize_url` MCP tool in the same sibling repo: builds a shareable
  link into QuEra's Gemini Logical Circuit Builder
  (`gemini-logical-circuit-builder.vercel.app`) for a small circuit — pure
  URL string encoding, no network call, no dependency on `stim` or Tsim.
  Rejects out-of-range qubit targets with `ok=False` rather than encoding a
  malformed link.
  Live capability check: 5 prompts (`eval/tsim_prompts.json`) run via real
  `chatqec ask` calls, all correct once past the off-topic classifier (2 of
  the 5, worded exactly as in the eval file, were classified off-topic
  before reaching the tool — see the capability-check doc). See
  [`docs/EXAMPLES.md`](docs/EXAMPLES.md#tsim--glcb-via-tsim_simulate--glcb_visualize_url)
  and [`docs/tsim_glcb_capability_check.md`](docs/tsim_glcb_capability_check.md).
- Tsim knowledge-base ingestion: `arxiv:2604.01059` (24 chunks, full LaTeX
  source of the Tsim paper), `github:QuEraComputing/tsim@21757967` (6 chunks,
  `docs/`), and 15 mintlify documentation pages (`tutorials/`,
  `api-reference/`, incl. `noise/` and `utils/` subpages) ingested and
  embedded — see
  [`docs/EXAMPLES.md`](docs/EXAMPLES.md#tsim-knowledge-base-paper--github-docs--mintlify-docs-ingested-2026-07-28).
  `eval/tsim_prompts.json` adds a 20-prompt benchmark set across 10
  categories (Clifford baseline, non-Clifford, magic-state prep, Stim/Tsim
  reference-convention differences, edge cases for both new tools, combined
  simulate+visualize, and corpus retrieval).
- Full-text arXiv ingestion via `chatqec ingest --arxiv <id> --full-text`:
  - LaTeX-source path (`ingestion/latex.py`, `fetchers/arxiv_source.py`) — cleans
    source while preserving math/code environments verbatim, keeps figure
    captions, drops images; resolves `\input`/`\include`.
  - Isolated-Marker PDF fallback (`fetchers/pdf_marker.py`) via `uvx` for
    PDF-only papers, GPU-enabled with `TORCH_DEVICE=cuda`.
  - `scripts/reingest_arxiv_fulltext.sh` for reproducible batch re-ingestion.
- Auto-load of `.env` in the CLI and explicit Gemini API-key configuration.
- Working Stim tool use: real stdio MCP client (`conversation/mcp_stdio.py`,
  replacing the stub) over the `mcp` SDK; the model decides when to call
  `stim_simulate`/`stim_diagram` and reports the tool's real output.
- Anthropic-shaped Gemini provider (`providers/gemini.py`) with tool-use
  translation, added to the `auto` fallback chain — Stim answers fail over from
  Claude to Gemini on credit/auth errors. `--mcp-server` now accepts the
  `anthropic`, `gemini`, or `auto` provider.
- Image input (multimodal prompts): attach up to 3 images in the Streamlit chat
  (`st.chat_input` file attach) or `chatqec ask --image <path>`; the answering VLM
  sees the figure and a Gemini-Flash caption augments retrieval. Fallback chain
  Anthropic→Gemini→Llama-4 (HF) all carry images; images are transient (never
  persisted). New `ImageAttachment`, `conversation/vision.py`, `ingestion/images.py`.
- Force-injection of matched canonical pages into the synthesizer context
  (`api.py`): definition/comparison queries now cite `canonical:<slug>` pages
  (previously `canonical_chunks=[]` meant they were matched but never used).
- Switchable verbose pipeline tracing (`chatqec/trace.py`): records every step
  (classify, rewrite, retrieve + per-chunk, synthesize, provider/model
  selection, tool-loop and MCP calls) with full payloads. `chatqec ask -v` /
  `chat -v` write a numbered, timestamped trace to
  `logs/chatqec-trace-<ts>.log` (override with `--log-file`), keeping the
  terminal to a single pointer line; the Streamlit app exposes a sidebar
  "Verbose trace" toggle that renders the trace inline.

### Changed
- Re-ingested the 13 foundational arXiv papers as full text (≈711 chunks vs. 13
  abstract chunks); corpus grew to ~820 embedded chunks.
- Cross-encoder reranker default changed from `BAAI/bge-reranker-base` to
  `mixedbread-ai/mxbai-rerank-base-v1` (Apache-2.0) to avoid China-origin
  models; `models.reranker` in `conf/config.yaml` is now honored by
  `_make_reranker()` (previously ignored).

### Removed
- GitLab CI pipeline (`.gitlab-ci.yml`); quality gates run via local pre-commit.

## [0.1.0] - 2026-06-10

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
- Sibling repo `chatqec-mcp-tools` (separate project): subprocess+rlimits sandbox runner; `stim_simulate` and `stim_diagram` tools; MCP server entry over stdio
- `MCPAdapter` translating MCP tool definitions to Anthropic-SDK schemas
- Anthropic-native `run_tool_loop` with iteration cap and unknown-tool fallback
- `Synthesizer.mcp_adapter` enables tool use; `SynthesisResult.tool_invocations` exposes per-call I/O
- `Answer.tool_calls` propagates Stim runs to library users
- CLI: `chatqec ask --mcp-server "chatqec-mcp-tools"` (and `chatqec chat --mcp-server …`) enables the tool loop end-to-end
- Stub `StdioMCPClient` placeholder (full async stdio client lands as a follow-up patch)
- Guards (spec §12):
  - `RateLimiter` (SQLite-backed) for per-IP per-action caps (20 asks/day, 5 tools/hour)
  - `CostBudget` (SQLite-backed) for daily USD-equivalent cap with graceful "service degraded" fallback
  - `HCaptchaVerifier` (free tier) with disabled-on-empty-secret pass-through for dev
  - Token-cost estimator (`estimate_cost_usd`) per spec §12.3 pricing table
- Streamlit app (`chatqec_app/`):
  - 4 pages (Chat, Sources, Watchlist, About)
  - Streaming Chat with citation cards and tool-output rendering (Stim SVG inline, samples as JSON)
  - Sources page with type/status/tier filtering (pandas-free, plain list filtering)
  - Watchlist page with "Submit a source" GitLab-issue link
  - About page with corpus summary
- App testing via `streamlit.testing.v1.AppTest` (4 page smoke tests)
- Config: `limits:` and `captcha:` sections in `conf/config.yaml`
- Eval set:
  - 50 Q&A pair seed eval set covering 6 question subtypes (`eval/qa_set.jsonl`)
  - `EvalCase` schema (StrEnum subtype), JSONL loader, `recall_at_k`, `citation_accuracy`, LLM-judged `faithfulness`
  - `chatqec eval run` CLI emits per-subtype + overall metrics as JSON
- Watchlist automation:
  - `refresh_arxiv` + `refresh_github` (real implementations); `refresh_rss` is a follow-up stub
  - `chatqec watch refresh --apply` writes `pending_review` records to the registry
  - GitLab CI scheduled `watchlist-refresh` job opens a weekly MR with new candidates (requires `WATCHLIST_PUSH_TOKEN` CI variable and a cron schedule)
- Issue template `.gitlab/issue_templates/add_source.md` for community source proposals

### Released

This release closes out the v1 scope from the design spec.
