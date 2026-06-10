# ChatQEC

An open-source research assistant for **Quantum Error Correction**.

Ask QEC questions; get cited answers grounded in a curated corpus of papers,
textbooks, GitHub repos, and lecture transcripts. Run small [Stim] circuits in
a sandbox. Use it as a Python library, a CLI, or a Streamlit web app.

[Stim]: https://github.com/quantumlib/Stim

> **Status: v0.1.0 — first public release.** The pipeline is end-to-end and
> testable; the public corpus is intentionally seeded with 5 canonical wiki
> pages and grows via the watchlist refresh job. See [Roadmap](#roadmap).

---

## Why

Existing chat assistants hallucinate on QEC: they confuse code distances,
misquote thresholds, fabricate paper titles, and can't run a stabilizer
circuit to check their own claim. ChatQEC is built specifically for QEC, with:

- **A curated wiki layer** of hand-reviewed canonical pages (surface code,
  stabilizer formalism, CSS codes, threshold theorem, magic state distillation)
  that bypass the retrieval ranking for definitional questions — if you ask
  "what is the surface code?", you always get the canonical page first.
- **Hybrid retrieval** (dense vectors + BM25 + cross-encoder reranking) over
  a typed source registry, with authority tiers and recency in the scoring.
- **Inline citations** to source IDs the reader can click and verify.
- **Sandboxed Stim** via a separate MCP tool server, so circuit-level
  questions get a real simulator answer — not a guess.
- **Honest refusals** when the evidence isn't in the corpus, instead of
  inventing a plausible-sounding one.

---

## Quick start

### Try it — CLI (`chatqec ask`)

```bash
git clone https://code.ornl.gov/qsc-as/chatqec
cd chatqec
uv sync --extra dev

# bring up Qdrant for the vector store
docker compose up -d

# set API keys (or copy to .env)
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=...

# ask
uv run chatqec ask "What is the surface code threshold under depolarizing noise?"
```

### Run the Streamlit app

```bash
uv run streamlit run chatqec_app/streamlit_app.py
```

Four pages: **Chat** (streaming Q&A with citation cards), **Sources**
(filterable registry view), **Watchlist** (what we track), **About**.

### Enable Stim execution

Install the sibling MCP server and point `--mcp-server` at it:

```bash
uv tool install chatqec-mcp-tools
uv run chatqec ask --mcp-server chatqec-mcp-tools \
  "Run 'H 0 \n CNOT 0 1 \n M 0 1' for 1000 shots — what's the sample distribution?"
```

The Stim tools (`stim_simulate`, `stim_diagram`) run in a subprocess sandbox
with memory and wall-clock limits.

---

## What to expect on your first run

**The corpus is intentionally small at v0.1.0.** The source registry ships
empty; only the 5 canonical wiki pages are pre-loaded. Definitional questions
on these topics will return cited answers immediately:

- Surface code
- Stabilizer formalism
- CSS codes
- Threshold theorem
- Magic state distillation

Anything outside that scope will return an honest refusal — not a
hallucinated answer. To get broader coverage, ingest sources via
`chatqec ingest` (or wait for the watchlist refresh) before asking.

**Costs.** A typical single-turn query costs roughly $0.01–$0.03 (one Haiku
classification + one Haiku rewrite + one Sonnet synthesis, all uncached).
Streaming and tool-use add a small overhead. The `chatqec.guards.CostBudget`
caps daily spend; the default is $5/day and is configurable via
`CHATQEC_DAILY_BUDGET_USD`.

**Anthropic is required.** There is no fully-local path. Gemini embeddings
can be swapped for local BGE/Nomic (`embedding.provider=local`), but Haiku
(classifier + rewriter) and Sonnet (synthesizer + judge) need an Anthropic
key. A self-host-only-on-open-weights path is out of scope for v1.

---

## What's in the box

| Layer | Module | Notes |
|---|---|---|
| Canonical wiki | [`knowledge/canonical/`](knowledge/canonical/) | 5 P0 pages; 52-page roster in [`_roster.yaml`](knowledge/canonical/_roster.yaml) |
| Source registry | [`sources/registry.jsonl`](sources/registry.jsonl) | Typed JSONL; lifecycle states (`active`/`superseded`/`retracted`/`stale`/`pending_review`) |
| Watchlist | [`sources/watchlist.yaml`](sources/watchlist.yaml) | arXiv categories, GitHub repos, YouTube channels |
| Ingestion | [`src/chatqec/ingestion/`](src/chatqec/ingestion/) | Fetchers (arXiv / GitHub / YouTube / URL / PDF), math-aware chunker, title-and-author resolver |
| Embedding | [`src/chatqec/embedding/`](src/chatqec/embedding/) | Gemini (default) or local BGE/Nomic; BM25 via fastembed |
| Store | [`src/chatqec/store/qdrant.py`](src/chatqec/store/qdrant.py) | Hybrid dense + sparse search |
| Retrieval | [`src/chatqec/retrieval/`](src/chatqec/retrieval/) | Spec §8.5 weighted scoring (`0.5·dense + 0.3·bm25 + 0.15·tier + 0.05·recency`), BGE cross-encoder reranker, canonical force-injection |
| Conversation | [`src/chatqec/conversation/`](src/chatqec/conversation/) | Haiku off-topic classifier + query rewriter; Sonnet synthesizer with native tool-use loop |
| Guards | [`src/chatqec/guards/`](src/chatqec/guards/) | Per-IP rate limit, daily USD budget, hCaptcha verifier |
| Eval | [`src/chatqec/eval/`](src/chatqec/eval/) + [`eval/`](eval/) | 50-pair Q&A set across 6 subtypes; recall@K + citation accuracy + LLM-judge faithfulness |
| MCP tools | [`chatqec-mcp-tools`](https://code.ornl.gov/qsc-as/chatqec-mcp-tools) | `stim_simulate`, `stim_diagram` over MCP stdio |

---

## How it works

```
                       ┌──────────────┐
question  ──▶ Haiku ──▶│  off-topic?  │── no ──▶ refuse politely
                       └──────┬───────┘
                              │ yes
                       ┌──────▼───────┐
                       │   Haiku      │  resolves pronouns,
                       │   rewriter   │  emits 2–3 search variants
                       └──────┬───────┘
                              │
        ┌─────────────────────▼──────────────────────┐
        │      Hybrid retriever (Qdrant + BM25)      │
        │ → spec §8.5 scoring → BGE reranker → top-K │
        │     ▲                                       │
        │     │  canonical wiki force-inject          │
        └─────┼───────────────────────────────────────┘
              │
       ┌──────▼─────────┐         ┌─────────────────────┐
       │ Sonnet         │  tools  │  MCP adapter        │
       │ synthesizer    │ ◀────▶ │  → stim_simulate    │
       │ (cited answer) │         │  → stim_diagram     │
       └────────────────┘         └─────────────────────┘
```

Full design rationale lives in
[`docs/superpowers/specs/2026-05-29-chatqec-design.md`](docs/superpowers/specs/2026-05-29-chatqec-design.md).

---

## Configuration

ChatQEC uses Hydra + OmegaConf. The root config is
[`conf/config.yaml`](conf/config.yaml); override anything from the command line:

```bash
uv run chatqec ask "..." --mcp-server chatqec-mcp-tools
CHATQEC_DAILY_BUDGET_USD=2.50 uv run streamlit run chatqec_app/streamlit_app.py
```

Required environment:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Haiku (classifier/rewriter) + Sonnet (synthesizer + judge) |
| `GEMINI_API_KEY` | `text-embedding-004` for dense vectors (or set `embedding.provider=local`) |
| `HCAPTCHA_SECRET` | Optional. Public Streamlit deploys; empty = captcha disabled |
| `CHATQEC_DAILY_BUDGET_USD` | Optional. Daily USD cap for the budget guard |
| `WATCHLIST_PUSH_TOKEN` | CI-only. GitLab token for the scheduled watchlist refresh MR |

---

## Eval

```bash
uv run chatqec eval run \
  --cases eval/qa_set.jsonl \
  --output eval/results.json
```

The seed set has 50 Q&A pairs across six subtypes:

| Subtype | Cases |
|---|---|
| definition | 8 |
| paper_lookup | 8 |
| comparison | 9 |
| math_detail | 9 |
| recent_niche | 8 |
| code_execution | 8 |

Metrics: **recall@K** (did the retriever surface a relevant source?),
**citation accuracy** (did the answer's citations point to the right sources?),
and **faithfulness** (LLM-judged: do the answer's claims follow from the
retrieved evidence?). Run output is JSON with per-subtype + overall scores.

> Headline numbers will land once the corpus is seeded. The pipeline is
> end-to-end testable now; meaningful eval scores need real content.

---

## Roadmap

**v0.1.0 (this release):**
- ✅ End-to-end pipeline: classifier → rewriter → hybrid retrieval → synthesizer
- ✅ Canonical wiki layer (5 P0 pages; 52-page roster planned)
- ✅ Sandboxed Stim execution via MCP sibling repo
- ✅ Streamlit app with per-IP rate limit, daily budget, hCaptcha
- ✅ 50-pair eval set + `chatqec eval run`
- ✅ Watchlist refresh (arXiv + GitHub) with scheduled CI MR

**v0.2 — content & quality:**
- Populate the source registry to ~500 documents (seed via watchlist + bulk arXiv)
- Build out the canonical wiki to the full 52-page roster
- Publish baseline eval numbers and the methodology behind them
- Hosted Streamlit demo

**v0.3 — tooling:**
- PyMatching decoder tool
- qiskit-qec tool (in the MCP sibling)
- Threshold-sweep helper
- Async stdio MCP client (current `StdioMCPClient` is a stub)

**v1.0 — production:**
- Real `from_config()` deploy flow with secret management
- Docker-isolated sandbox (current uses subprocess + rlimits)
- Per-source retraction propagation

---

## Contributing

**Propose a source** for the corpus: open an issue using the
[`add_source` template](.gitlab/issue_templates/add_source.md). Maintainer
reviews license + relevance, then `chatqec ingest` plus the weekly watchlist
refresh keeps it current.

**Work on the code:**

```bash
uv sync --extra dev
uv run pytest                          # v0.1.0: 231 tests, 89% line coverage
uv run ruff check src tests chatqec_app
uv run mypy src
uv run chatqec canonical validate
```

Conventional Commits + small, reviewable diffs. The design spec
([`docs/superpowers/specs/`](docs/superpowers/specs/)) and implementation plan
([`docs/superpowers/plans/`](docs/superpowers/plans/)) are the source of
truth — extend them before adding a feature outside the v1 scope.

---

## Citing

If ChatQEC helps your research, please cite it as:

```bibtex
@software{chatqec_2026,
  author  = {Afrose, Sharmin},
  title   = {ChatQEC: an open-source research assistant for Quantum Error Correction},
  version = {0.1.0},
  year    = {2026},
  url     = {https://code.ornl.gov/qsc-as/chatqec}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE). The sibling MCP server
[`chatqec-mcp-tools`](https://code.ornl.gov/qsc-as/chatqec-mcp-tools) is also
Apache 2.0.
