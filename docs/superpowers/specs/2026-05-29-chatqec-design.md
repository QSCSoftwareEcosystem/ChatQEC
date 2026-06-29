# ChatQEC — Design Document

| Field | Value |
|---|---|
| Date | 2026-05-29 |
| Status | Draft, awaiting approval |
| Author | sharmin.afrose.vt@gmail.com |
| Scope | v1 architecture + v2 roadmap |
| Next step on approval | Invoke `superpowers:writing-plans` to produce an implementation plan |

---

## 1. Problem statement

Build **ChatQEC**, an open-source semi-automated research assistant that answers questions about Quantum Error Correction (QEC) by retrieving from a heterogeneous, continuously growing corpus (research papers, textbooks, lecture notes, GitHub repos, YouTube transcripts, blogs, online courses), with citations, and that can execute small QEC simulations on demand.

Two deployable surfaces:
- A pip-installable Python library + CLI that anyone can self-host with their own Anthropic + Gemini API keys.
- A public Streamlit web app at the project's GitHub Pages / Hugging Face Space that anyone can use without setup.

## 2. Non-goals (v1)

- Not a general-purpose quantum computing assistant — QEC focus is enforced via off-topic classification.
- Not a quantum simulator front-end for arbitrary user code — sandboxed Stim execution only.
- Not a substitute for primary sources — answers must cite, and complex theory questions should point to the textbook chapter rather than invent.
- Not a multi-user collaborative platform — sessions are anonymous and per-tab.

## 3. User personas

| Persona | Need | Example query |
|---|---|---|
| Graduate student starting on QEC | Quick conceptual orientation with primary-source pointers | "What is the difference between the surface code and the color code?" |
| Researcher mid-project | Sanity-check definitions, find related work, run a small Stim circuit | "What's the threshold for the rotated surface code under SD6 noise? Run d=3,5,7 for me." |
| Engineer evaluating QEC stacks | Compare libraries and decoder options | "What decoders are open-source and which papers benchmark them on Stim circuits?" |
| Course participant | Tutorial-style explanations with worked examples | "Explain stabilizer measurement for the [[7,1,3]] code, step by step." |

## 4. High-level architecture

```
┌─ INGESTION LAYER (offline / scheduled) ──────────────────────────────┐
│                                                                      │
│   Watchlist (YAML)                Canonical wiki (knowledge/canonical/)│
│        │                                       │                     │
│   Fetchers → Parsers → Normalizer              Frontmatter parser    │
│   (arxiv, github, youtube, web, pdf)                 │               │
│        │                                             │               │
│        └──────────────┬──────────────────────────────┘               │
│                       ▼                                              │
│                  Chunker (math-aware / structural)                   │
│                       │                                              │
│              ┌────────┴───────┐                                      │
│              ▼                ▼                                      │
│   Embedder (Gemini)     Registry writer (sources/registry.jsonl)     │
│              │                                                       │
│              ▼                                                       │
│              Qdrant (vector + BM25 sparse, single collection)        │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─ RETRIEVAL + REASONING (online, per query) ──────────────────────────┐
│                                                                      │
│   Query → Off-topic classifier (Haiku) → Rewriter (Haiku, 3 variants)│
│                                                  │                   │
│                                                  ▼                   │
│              Hybrid retrieval (Qdrant: vector + BM25) + RRF          │
│                                                  │                   │
│                                                  ▼                   │
│              Cross-encoder reranker (BGE, local CPU)                 │
│                                                  │                   │
│                                                  ▼                   │
│              Canonical force-injection (topic_slug match)            │
│                                                  │                   │
│                                                  ▼                   │
│              Synthesizer (Claude Sonnet, native tool-use loop)       │
│                          │              │                            │
│                          │              ▼                            │
│                          │     MCP adapter → chatqec-mcp-tools       │
│                          │     (Stim sandbox in v1)                  │
│                          ▼                                           │
│              Final answer + inline citations + tool outputs          │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─ SURFACES ───────────────────────────────────────────────────────────┐
│   • Python library `chatqec`                                         │
│   • CLI `chatqec ask / chat / ingest / sources / watch / compact`    │
│   • Streamlit app (Chat, Sources, Watchlist, About)                  │
│   • (optional) FastAPI service for programmatic access               │
└──────────────────────────────────────────────────────────────────────┘
```

## 5. Knowledge store

### 5.1 Choice: hybrid RAG on Qdrant

Evaluated alternatives in brainstorming: pure LLM-Wiki (doesn't scale past a few hundred pages), pure vector RAG (loses exact-terminology matches, struggles with math), GraphRAG (overkill for v1), Gemini long-context only (lost-in-the-middle, no clean citations), agentic multi-retriever (premature). Chose **hybrid retrieval (BM25 + vector + cross-encoder reranker)** on Qdrant because:

1. Heterogeneous sources flow through one uniform pipeline.
2. BM25 catches exact terminology (`[[7,1,3]]`, `Steane code`, `BP+OSD`) that pure embeddings miss on math-heavy content.
3. Reranking lifts top-1 precision substantially on technical Q&A.
4. Qdrant supports vector + sparse + payload filtering in one store — no separate Elasticsearch.
5. Qdrant scales to the large/aspirational corpus target (1000s of sources) where ChromaDB would force a migration.

### 5.2 Canonical wiki layer

Hand-curated markdown pages in `knowledge/canonical/`, one per core QEC concept, with frontmatter:

```yaml
---
topic_slug: surface-code
title: "Surface Code"
aliases: ["toric code variant", "planar code"]
see_also: [stabilizer-formalism, color-code, mwpm-decoder, lattice-surgery]
authority_tier: canonical
last_reviewed: 2026-05-29
maintainers: ["sharmin"]
---
```

Chunked section-by-section, embedded into the same Qdrant collection with `authority_tier=canonical`, and **force-injected** into context whenever a query's detected topic matches the page's `topic_slug` or `aliases`.

The v1 roster of 52 canonical pages (P0/P1/P2 priority) is in Appendix A.1.

## 6. Source registry & lifecycle

The registry is the single source of truth for "what is in the knowledge base, what version, and what's still trusted." It lives in Git as **JSONL** (one record per line — clean diffs in PRs).

### Schema

```json
{
  "id": "arxiv:2408.13687v1",
  "type": "arxiv | github | youtube | blog | pdf | canonical_wiki",
  "url": "https://arxiv.org/abs/2408.13687",
  "title": "...",
  "authors": ["..."],
  "year": 2024,
  "version": "v1",
  "content_sha256": "ab12cd34...",
  "license": "arXiv non-exclusive",
  "authority_tier": "canonical | peer_reviewed | preprint | blog | video",
  "ingested_at": "...",
  "last_checked_at": "...",
  "status": "active | superseded | retracted | stale | pending_review",
  "supersedes": null,
  "superseded_by": null,
  "chunk_count": 47,
  "chunk_ids": ["..."],
  "verified": true,
  "notes": "..."
}
```

### Lifecycle states

| Status | Retrieval behavior |
|---|---|
| `active` | Returned normally |
| `superseded` | Excluded by default, returnable on explicit request |
| `retracted` | Excluded always; chunks kept for audit |
| `stale` | Returned with a "may be outdated" flag |
| `pending_review` | Not yet in live index — awaiting human PR approval |

No hard deletes in v1. `chatqec compact` physically purges `retracted` and `superseded` chunks older than 90 days when invoked manually.

### Per-source-type refresh logic

| Type | Refresh signal | Action |
|---|---|---|
| arXiv | New version number (v1→v2) | Re-ingest; mark old `superseded` |
| GitHub | `last_commit_sha` differs from HEAD | Diff file list; re-embed changed/added, mark deleted `retracted` |
| YouTube | Video 404 | `retracted`; otherwise transcripts stable |
| Blog/URL | `content_sha256` differs | Re-ingest; old → `superseded`. 404 → `retracted` |
| PDF (manual) | None (manual `chatqec re-ingest`) | User-driven |
| Canonical wiki | Git commit changes | Standard ingest re-run on PR merge |

## 7. Ingestion pipeline

### 7.1 Watchlist (`sources/watchlist.yaml`)

```yaml
arxiv:
  categories: [quant-ph]
  keywords: [surface code, stabilizer code, magic state, fault-tolerant,
             qLDPC, threshold, decoder, Floquet code, lattice surgery]
  authors: [Preskill, Gottesman, Kitaev, Terhal, Bombin, Litinski,
            Fowler, Yoder, Gidney, Bravyi, Cross, Tomita, Roffe]
github:
  - repo: quantumlib/Stim
  - repo: oscarhiggott/PyMatching
  - repo: qiskit-community/qiskit-qec
  # ... (see Appendix A.4)
youtube:
  channels: ["@GoogleQuantumAI", "@qiskit", "@Quantinuum",
             "@SimonsInstituteTOC", "@PerimeterInstitute"]
rss:
  - https://quantum-journal.org/feed/
  - https://research.ibm.com/blog/feed.xml
  - https://scottaaronson.blog/?feed=rss2
```

A weekly **GitHub Action** runs `chatqec watch refresh`:
1. Fetch candidates from each watchlist entry.
2. Deduplicate via `content_sha256`.
3. Write candidates as `status=pending_review` into `registry.jsonl`.
4. Open a PR titled `chore(ingest): N new sources for review`.
5. On merge, a second action embeds and flips status to `active`.

### 7.2 Fetchers, parsers, chunkers

| Source | Fetcher | Parser | Chunker |
|---|---|---|---|
| arXiv | LlamaIndex `ArxivReader` + arXiv API **search-by-title-and-author** (see §7.4) | `unstructured` PDF parser with math-protect | Math-aware, ~512-token chunks, 64-token overlap |
| GitHub | LlamaIndex `GithubRepositoryReader` (READMEs + docs/) | markdown-it for `.md`, AST-aware for code | One chunk per heading section for prose; per function/class for code |
| YouTube | `yt-dlp --write-auto-subs --skip-download` for transcripts | Custom timestamp-aware splitter | Time-segmented (~60s windows), `start_time` recorded |
| Blog / URL | `defuddle` (skill) for clean markdown extraction | markdown-it | Section-based on headings |
| PDF (manual) | Direct file upload | `unstructured[pdf]` | Math-aware |
| Canonical wiki | Local filesystem | Frontmatter + markdown | Section-based on `##` |

**Math protector:** chunker never splits inside `$…$`, `$$…$$`, `\begin{equation}…\end{equation}`, `\begin{align}…\end{align}`. This is the single biggest accuracy lever for QEC content.

### 7.3 Embedding and indexing

- **Default embedder:** Gemini `text-embedding-004` (768-dim). Cheap, long input context, free with Gemini API key.
- **Local fallback:** `nomic-embed-text-v1.5` for users self-hosting without a Gemini key.
- Same embedder must be used for corpus and query — adapter pattern with a hard assertion at startup.
- BM25 lives in the same Qdrant collection using sparse vectors — no separate keyword index.

### 7.4 Title-and-author search (mandatory pattern — see verification finding in Appendix A.10)

**Never trust an LLM-generated arXiv ID.** The verification pass over 84 LLM-claimed IDs found only 14% fully correct. All ingestion of LLM-suggested sources MUST go through a search step:

```python
def resolve_arxiv_paper(title_keywords: str, authors: list[str]) -> str | None:
    """Returns an arxiv id by searching, not by trusting an LLM-claimed id."""
    query = f'ti:"{title_keywords}" AND au:"{authors[0]}"'
    results = arxiv.search(query, max_results=5)
    # rank by title+author similarity, year proximity
    ...
```

A `chatqec ingest --resolve-by-title` command applies this to a list of (title, author) candidates and emits resolved IDs for review.

## 8. Retrieval pipeline

### 8.1 Query rewriting

Every query passes through **Claude Haiku** for:
1. Off-topic detection — if the query isn't QEC-related, refuse politely (cheap classifier, saves cost on public app).
2. Pronoun/reference resolution (multi-turn).
3. Abbreviation expansion (`MWPM` → also search for `minimum-weight perfect matching`).
4. Generation of 2–3 query variants for multi-query retrieval.
5. Topic detection (which `topic_slug`s to force-inject).

### 8.2 Hybrid retrieval + reciprocal rank fusion

For each query variant: parallel Qdrant calls for vector (k=30) and BM25 (k=30). Merge via RRF. Apply payload filter `status="active"` (default). Yield top 50 candidates.

### 8.3 Reranking

**Local cross-encoder** (`mixedbread-ai/mxbai-rerank-base-v1`) reranks the top 50 → top 8. Pure CPU, ~100–300 ms, free. (Replaced `BAAI/bge-reranker-base` to avoid China-origin models.)

### 8.4 Canonical force-injection

For every detected `topic_slug`, the canonical wiki page is added to context regardless of vector/BM25 score. This grounds the LLM in correct terminology.

### 8.5 Scoring formula

```
final_score = 0.5 · vector + 0.3 · bm25 + 0.15 · tier_weight + 0.05 · recency

tier_weight: canonical=1.0, peer_reviewed=0.9, preprint=0.7, blog=0.5, video=0.5
```

All four coefficients live in `conf/retriever.yaml` and are tunable.

## 9. Conversation & synthesis layer

### 9.1 Multi-turn handling

- In-session message history lives in Streamlit session state (in-memory, 30-min TTL).
- The rewriter receives the last N turns and produces a self-contained rewritten query.
- A "same-topic follow-up?" flag lets us reuse the previous turn's retrieved chunks as candidates (cache hit) plus a small fresh retrieval.
- Cited sources for the session are persistent in browser localStorage.
- **No server-side per-user accounts in v1.** Sessions are anonymous.

### 9.2 Anthropic + Gemini split (final)

| Component | Model | Reason |
|---|---|---|
| Off-topic classification + query rewriting | Claude Haiku 4.5 | Fast, cheap |
| Embeddings | Gemini `text-embedding-004` | Cheap, long input window |
| Reranking | local `mxbai-rerank-base-v1` | Free, fast on CPU; non-China origin |
| Synthesis | Claude Sonnet 4.6 | Strong reasoning, native tool-use, good citation discipline |
| Long-context fallback (low-confidence path) | Gemini 2.5 Pro | 1M context for broader source slice when retrieval is sparse |
| (Optional) self-grader | Claude Haiku 4.5 | "Does the answer actually use the cited evidence?" |

**Prompt caching:** the system prompt + canonical pages that are always-injected for a session use Anthropic's prompt-cache feature (5-min TTL) → ~70% input-token savings on repeated queries in a session.

### 9.3 Tool-use loop

The synthesizer uses native Anthropic SDK tool-use:

```python
while True:
    response = anthropic.messages.create(
        model="claude-sonnet-4-6",
        messages=conversation,
        tools=mcp_adapter.tool_schemas(),
        system=system_prompt,
    )
    if response.stop_reason != "tool_use":
        break
    for tool_use in extract_tool_uses(response):
        result = mcp_adapter.invoke(tool_use.name, tool_use.input)
        conversation.append({"role": "assistant", "content": response.content})
        conversation.append({"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": tool_use.id, "content": result
        }]})
```

Hard caps per query: 5 tool-use iterations, 30 s wall clock, 12 chunks of context, 4000 output tokens.

## 10. MCP tool server (`chatqec-mcp-tools`)

Lives as a **separate Git repo** so it can be installed by any MCP-capable client (Claude Desktop, Claude Code, third-party apps) independently of ChatQEC.

### 10.1 v1 tools

| Tool | Backed by | Covers |
|---|---|---|
| `stim_simulate` | Stim | Run user-supplied Stim circuit → measurement samples, detection events, observable flips |
| `stim_diagram` | Stim + matplotlib | Render circuit as SVG; validates parseability |

Sandbox: each invocation in a fresh Docker container, 30 s wall clock, 1 GB memory, 1 CPU, no network egress, read-only filesystem except `/tmp`. Output capped at 100 KB.

### 10.2 v2 tools (roadmap)

| Tool | Backed by | Covers |
|---|---|---|
| `pymatching_decode` | PyMatching | MWPM decoding |
| `belief_propagation_decode` | quantumgizmos/ldpc | BP / BP+OSD for qLDPC |
| `qiskit_qec_run` | qiskit-qec | Run small Qiskit QEC circuits |
| `clifford_simulate` | Stim tableau | Stabilizer formalism state output |
| `stabilizer_check` | Stim tableau | Commutation checks, logical operator finder |
| `code_search` | Error Correction Zoo data + curated table | Look up codes by [[n,k,d]] or family |
| `threshold_sweep` | Stim + PyMatching | Parametric distance × noise sweeps |
| `logical_error_curve` | matplotlib | Plot logical error vs physical error |
| `monte_carlo_estimate` | Stim sampler | Logical failure rate to target precision |
| `decode_compare` | PyMatching + others | Multi-decoder comparison on same syndromes |

### 10.3 Integration mechanism

Hybrid: **native Anthropic tool-use** is used by the synthesizer for low latency; the **MCP server** is where tools are *implemented*. A thin adapter in `chatqec` discovers the MCP server's tool list at startup and registers them as native tools. This means:

- Anyone with Claude Desktop or Claude Code can install `chatqec-mcp-tools` and chat with QEC tools without our Streamlit app.
- Our hosted Streamlit app gets native-API latency (no extra hop on the user-facing path).
- New tools added to the MCP server appear automatically in the chat — no main-app change.

## 11. Surfaces

### 11.1 Python library

```python
from chatqec import ChatQEC

bot = ChatQEC.from_config("config.yaml")
answer = bot.ask("What's the threshold of the toric code under depolarizing noise?")
print(answer.text)
print(answer.citations)   # list of source registry entries
print(answer.tool_calls)  # list of Stim invocations + outputs
print(answer.confidence)  # "high" | "medium" | "low"

# Streaming
for chunk in bot.ask_stream("..."):
    print(chunk.delta, end="")
```

### 11.2 CLI

```bash
chatqec init                          # configure keys, install MCP server
chatqec ask "your question"           # one-shot Q&A
chatqec chat                          # interactive REPL with history
chatqec ingest --pdf paper.pdf
chatqec ingest --arxiv 2408.13687     # verified by API on every call
chatqec ingest --resolve-by-title "surface code below threshold" --author "Google Quantum AI"
chatqec ingest --github quantumlib/Stim
chatqec ingest --youtube <url>
chatqec watch refresh                 # run watchlist pipeline manually
chatqec sources list                  # registry table
chatqec sources show <id>             # detail
chatqec sources mark-retracted <id>   # lifecycle override
chatqec compact                       # purge retracted/superseded > 90d
chatqec eval run                      # run eval set, print metrics
```

### 11.3 Streamlit app (`chatqec_app/`)

Pages:
1. **Chat** — streaming, inline citation chips that expand to source cards, tool outputs rendered (Stim diagrams as SVG, sampled measurements as tables).
2. **Sources** — searchable, filterable table of `registry.jsonl`. Filter by type / year / authority tier / status. Click a row → full metadata + which chunks are in the index.
3. **Watchlist** — read-only view of `watchlist.yaml`. "Submit a source" button opens a GitHub Issue with prefilled template.
4. **About** — what's in the corpus, how it works, GitHub link, citation.

Sidebar controls: model selector (Sonnet/Haiku), authority tier filter, "include superseded sources" toggle.

Footer: token count + estimated cost for current session (transparency).

## 12. Public-facing concerns

### 12.1 Rate limits

| Limit | Cap |
|---|---|
| Anonymous public users | 20 questions / IP / day |
| Tool execution (Stim) | 5 invocations / IP / hour |
| Per-question wall clock | 30 s |
| Per-question tool iterations | 5 |
| Per-question retrieved chunks | 12 |
| Per-question output tokens | 4000 |

Implemented via `slowapi` middleware backed by SQLite (`chatqec.db`). Resets at UTC midnight.

### 12.2 Sandbox safety

Already detailed in §10.1: Docker container per invocation, no network, time/memory/CPU bounded, output capped.

CAPTCHA (hCaptcha free tier) gates the first question per session on the public app.

Off-topic queries detected by the Haiku classifier are refused before reaching Sonnet — saves both cost and abuse vector.

### 12.3 Cost model (rough, per query, no tool use)

| Cost | Per query |
|---|---|
| Off-topic classify + rewrite (Haiku) | ~$0.0005 |
| Embedding (Gemini) | ~$0.00001 |
| Reranking | $0 (local) |
| Synthesis (Sonnet, with prompt cache) | ~$0.015 |
| Tool use (when invoked) | ~$0.001 sandbox |
| **Total** | **~$0.015–0.020** |

Daily budget cap enforced via env var. When hit, public app degrades to "service temporarily unavailable; try the self-host instructions."

## 13. Testing strategy

| Layer | Tests | Tooling |
|---|---|---|
| Unit | Parsers, math-aware chunker (golden tests), registry CRUD, watchlist diff | pytest |
| Integration | Ingest end-to-end on fixture corpus (~10 sources); retrieval returns expected chunks | pytest + Qdrant Docker |
| Eval | Curated 50-pair Q&A set (drawn from the canonical wiki topics); metrics: recall@10, citation accuracy, answer faithfulness (LLM judge) | `chatqec eval` |
| Tool | Each MCP tool has golden input/output cases | pytest in `chatqec-mcp-tools` |
| Smoke | `chatqec ask "What is the surface code?"` against frozen corpus matches expected pattern | CI on every PR |

Coverage target: ≥80% on `chatqec` library code; lower bar for Streamlit UI.

## 14. Repo layout

```
chatqec/
├── pyproject.toml              # uv, Python ≥3.11
├── README.md
├── LICENSE                     # Apache-2.0
├── .github/workflows/
│   ├── ci.yml
│   ├── watchlist-refresh.yml   # weekly
│   └── publish.yml
├── conf/                       # Hydra configs
│   ├── config.yaml
│   ├── retriever/
│   ├── synthesizer/
│   └── ingestion/
├── src/chatqec/
│   ├── __init__.py
│   ├── config.py               # frozen dataclasses
│   ├── api.py                  # ChatQEC class
│   ├── cli.py                  # Click-based
│   ├── ingestion/
│   │   ├── fetchers/           # arxiv, github, youtube, web, pdf
│   │   ├── parsers/
│   │   ├── normalizers/
│   │   ├── chunkers/           # math-aware
│   │   ├── embedder.py
│   │   ├── registry.py
│   │   ├── resolver.py         # title-and-author search (§7.4)
│   │   └── watchlist.py
│   ├── retrieval/
│   │   ├── qdrant_store.py
│   │   ├── hybrid.py
│   │   ├── reranker.py
│   │   └── canonical_injector.py
│   ├── conversation/
│   │   ├── synthesizer.py
│   │   ├── rewriter.py
│   │   ├── classifier.py       # off-topic detector
│   │   ├── memory.py
│   │   └── mcp_adapter.py
│   ├── eval/
│   │   ├── dataset.py
│   │   └── metrics.py
│   └── utils/
├── chatqec_app/                # Streamlit
│   ├── streamlit_app.py
│   ├── pages/
│   │   ├── 1_Chat.py
│   │   ├── 2_Sources.py
│   │   ├── 3_Watchlist.py
│   │   └── 4_About.py
│   └── components/
├── knowledge/
│   ├── canonical/              # 52 markdown pages (Appendix A.1)
│   └── README.md               # contribution guide
├── sources/
│   ├── registry.jsonl
│   └── watchlist.yaml
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
└── docs/
    ├── architecture.md
    ├── contributing.md
    └── superpowers/specs/      # this file lives here
```

Sibling repo:

```
chatqec-mcp-tools/
├── pyproject.toml
├── README.md
├── src/chatqec_mcp_tools/
│   ├── server.py
│   ├── tools/
│   │   ├── stim_simulate.py        # v1
│   │   ├── stim_diagram.py         # v1
│   │   ├── pymatching_decode.py    # v2
│   │   ├── qiskit_qec_run.py       # v2
│   │   ├── stabilizer_check.py     # v2
│   │   ├── code_search.py          # v2
│   │   ├── threshold_sweep.py      # v2
│   │   ├── logical_error_curve.py  # v2
│   │   ├── decode_compare.py       # v2
│   │   └── monte_carlo_estimate.py # v2
│   └── sandbox/
│       ├── docker_runner.py
│       └── policies.py
└── tests/
```

## 15. Stack & framework choice

**No agent framework in v1.** Use native SDKs.

Rejected: CrewAI (overkill for single-agent + tools), LangChain/LangGraph (heavy abstraction, volatile API, hides debug-relevant detail), Haystack (extra DSL), PydanticAI (immature ecosystem).

Adopted:
- **Anthropic SDK** for Claude.
- **`google-generativeai`** for Gemini embeddings + long-context fallback.
- **`qdrant-client`** for the store.
- **`sentence-transformers`** for the BGE reranker.
- **`mcp` Python SDK** for the tool server.
- **LlamaIndex *readers only*** (`llama_index.readers.arxiv`, `.github`, `.youtube_transcript`, `.file.PDFReader`) — used as one-off `reader.load_data(...)` calls inside our fetchers. We never import LlamaIndex core or its agents.
- **Streamlit, Click, Hydra + OmegaConf, Pydantic, slowapi, tenacity, unstructured[pdf], defuddle.**

```toml
# pyproject.toml dependencies (excerpt)
[project]
dependencies = [
  "anthropic>=0.40",
  "google-generativeai>=0.8",
  "qdrant-client>=1.12",
  "sentence-transformers>=3.0",
  "llama-index-readers-arxiv",
  "llama-index-readers-github",
  "llama-index-readers-youtube-transcript",
  "llama-index-readers-file",
  "defuddle",
  "unstructured[pdf]",
  "streamlit>=1.40",
  "click>=8.1",
  "hydra-core>=1.3",
  "omegaconf>=2.3",
  "mcp>=0.9",
  "pydantic>=2.8",
  "slowapi",
  "tenacity",
  "arxiv",  # for the title-and-author resolver
]
```

We'll revisit framework choice for **v2+ multi-agent** (LangGraph or PydanticAI), but not before.

## 16. v1 scope summary & v2+ roadmap

### v1 (target: 3–4 weeks of focused work)

- Hybrid RAG (Qdrant + BM25 + cross-encoder reranker)
- Canonical wiki layer (52 hand-written pages, roster in Appendix A.1)
- Source registry with full lifecycle states (JSONL in Git)
- Watchlist + weekly GitHub Action refresh
- Ingestion for arXiv, GitHub, YouTube transcripts, blogs/URLs, manual PDFs
- **Title-and-author search resolver** (§7.4) — required by the verification finding
- Off-topic classification + multi-turn chat with query rewriting
- Stim execution via MCP server (`stim_simulate`, `stim_diagram`)
- Anthropic + Gemini split as specified in §9.2
- Streamlit public app + Python library + CLI
- Rate limits, sandboxing, abuse guardrails
- Eval set (~50 Q&A pairs) + metrics
- Tests + CI

### v2

- MCP tool expansion (PyMatching, qiskit-qec, threshold sweep, decoder comparison, logical-error-curve, stabilizer-check, code-search, Monte Carlo)
- Title-search-driven backfill of the unverified candidate paper list (Appendix A.3) into the registry
- Eval set growth to 200+ pairs spanning all six question subtypes (definition, paper lookup, comparison, mathematical detail, recent/niche, code-execution)
- Confidence-based fallback to Gemini long-context

### v2.5 / v3

- GraphRAG / KG-augmented retrieval for multi-hop questions
- Multi-agent decomposition (query → specialist retrievers → synthesizer)
- Dynamic source fetching as tools (`arxiv_fetch_paper`, `github_fetch_file`)
- User accounts / persistent server-side conversation memory
- Revisit framework choice (LangGraph / PydanticAI)

## 17. Open questions & risks

| # | Risk / question | Mitigation |
|---|---|---|
| 1 | LLM-suggested arXiv IDs are unreliable (verification finding) | Mandatory title-and-author search; never trust a raw LLM-claimed ID |
| 2 | Math content garbled by naive chunking | Math-protector chunker; golden tests on equations |
| 3 | Public Streamlit cost-runaway | Hard daily budget cap with graceful degradation |
| 4 | Stim sandbox abuse (long-running simulations) | Wall-clock + memory + CPU + output caps; per-IP tool-use cap |
| 5 | Textbook licensing (Nielsen & Chuang, Lidar & Brun) | Link-only, never ingest; canonical wiki summaries with page citations |
| 6 | Authority-tier weighting needs tuning | Coefficients in `conf/retriever.yaml`; tunable via eval set |
| 7 | Blogs/RSS endpoint discovery (not done in research phase) | Manual seed list in v0.5 (Appendix A.7); contributor PRs thereafter |
| 8 | Watchlist may flood the registry with low-value papers | `pending_review` gate; PRs require maintainer approval |

---

# Appendix A — Seed Corpus Inventory

The inventory was assembled via 7 parallel research agents during the brainstorming phase. Six agents were offline-only (web tools denied in their environment) and produced memory-based drafts; one agent (canonical-topics) needed no web access and produced production-ready output. A subsequent verification pass against the arXiv API exposed that LLM-generated IDs are largely unreliable — see Appendix A.10.

## A.1 Canonical wiki page roster (52 pages, production-ready)

This roster was produced by the canonical-topics research agent and is the v1 page list to be authored under `knowledge/canonical/`. Priorities are P0 (must-have), P1 (high), P2 (nice-to-have).

### P0 — Foundations (must-have for v1 launch)

```yaml
- topic_slug: "pauli-group"
  title: "Pauli Group"
  aliases: ["Pauli operators", "n-qubit Pauli group"]
  see_also: ["stabilizer-formalism", "pauli-channels", "logical-operators"]
  bucket: "foundations"
- topic_slug: "stabilizer-formalism"
  title: "Stabilizer Formalism"
  aliases: ["stabilizer codes intro", "Pauli stabilizers", "Gottesman formalism"]
  see_also: ["pauli-group", "css-codes", "logical-operators", "syndrome-extraction", "clifford-group"]
- topic_slug: "nkd-notation"
  title: "[[n,k,d]] Code Parameters"
  aliases: ["code distance", "n k d notation", "code parameters"]
  see_also: ["logical-operators", "stabilizer-formalism", "threshold-theorem"]
- topic_slug: "logical-operators"
  title: "Logical Operators"
  aliases: ["logical Pauli operators", "logical X", "logical Z", "encoded operators"]
  see_also: ["stabilizer-formalism", "nkd-notation", "transversal-gates"]
- topic_slug: "knill-laflamme"
  title: "Knill-Laflamme Conditions"
  aliases: ["KL conditions", "QEC conditions"]
  see_also: ["stabilizer-formalism", "nkd-notation", "syndrome-extraction"]
- topic_slug: "syndrome-extraction"
  title: "Syndrome Extraction"
  aliases: ["stabilizer measurement", "syndrome measurement", "parity checks"]
  see_also: ["stabilizer-formalism", "decoding", "fault-tolerance", "measurement-errors", "circuit-level-noise"]
- topic_slug: "decoding"
  title: "Decoding (General)"
  aliases: ["quantum decoder", "syndrome decoding"]
  see_also: ["syndrome-extraction", "mwpm-decoder", "bp-osd-decoder", "union-find-decoder", "threshold-theorem"]
- topic_slug: "threshold-theorem"
  title: "Threshold Theorem"
  aliases: ["fault tolerance threshold", "accuracy threshold theorem"]
  see_also: ["fault-tolerance", "concatenated-codes", "surface-code", "circuit-level-noise"]
- topic_slug: "fault-tolerance"
  title: "Fault Tolerance"
  aliases: ["fault-tolerant quantum computation", "FTQC"]
  see_also: ["threshold-theorem", "transversal-gates", "syndrome-extraction", "magic-state-distillation"]
- topic_slug: "clifford-group"
  title: "Clifford Group"
  aliases: ["Clifford gates", "Clifford operations"]
  see_also: ["pauli-group", "transversal-gates", "magic-state-distillation"]
- topic_slug: "shor-code"
  title: "Shor Code"
  aliases: ["9-qubit code", "Shor 9-qubit code", "[[9,1,3]] code"]
  see_also: ["steane-code", "css-codes", "concatenated-codes"]
- topic_slug: "steane-code"
  title: "Steane Code"
  aliases: ["[[7,1,3]] code", "7-qubit Steane code", "Hamming-based code"]
  see_also: ["css-codes", "shor-code", "transversal-gates", "color-code"]
- topic_slug: "css-codes"
  title: "CSS Codes"
  aliases: ["Calderbank-Shor-Steane codes", "CSS construction"]
  see_also: ["stabilizer-formalism", "steane-code", "surface-code", "qldpc-codes"]
- topic_slug: "five-qubit-code"
  title: "Five-Qubit Code"
  aliases: ["perfect code", "[[5,1,3]] code", "smallest QEC code"]
  see_also: ["stabilizer-formalism", "knill-laflamme", "nkd-notation"]
- topic_slug: "surface-code"
  title: "Surface Code"
  aliases: ["toric code", "planar surface code", "Kitaev surface code", "rotated surface code"]
  see_also: ["css-codes", "color-code", "mwpm-decoder", "lattice-surgery", "circuit-level-noise"]
- topic_slug: "color-code"
  title: "Color Code"
  aliases: ["2D color code", "triangular color code", "Bombin color code"]
  see_also: ["surface-code", "css-codes", "transversal-gates"]
- topic_slug: "bacon-shor-code"
  title: "Bacon-Shor Code"
  aliases: ["Bacon-Shor subsystem code", "compass code family"]
  see_also: ["subsystem-codes", "shor-code", "css-codes"]
- topic_slug: "transversal-gates"
  title: "Transversal Gates"
  aliases: ["transversal logic", "Eastin-Knill"]
  see_also: ["clifford-group", "steane-code", "reed-muller-codes", "magic-state-distillation"]
- topic_slug: "magic-state-distillation"
  title: "Magic State Distillation"
  aliases: ["MSD", "T-state distillation"]
  see_also: ["magic-state-injection", "triorthogonal-codes", "reed-muller-codes", "t-gate-count"]
- topic_slug: "magic-state-injection"
  title: "Magic State Injection"
  aliases: ["state injection", "T-state injection"]
  see_also: ["magic-state-distillation", "lattice-surgery", "transversal-gates"]
- topic_slug: "lattice-surgery"
  title: "Lattice Surgery"
  aliases: ["surface code lattice surgery", "merge and split"]
  see_also: ["surface-code", "code-deformation", "magic-state-injection"]
- topic_slug: "reed-muller-codes"
  title: "Reed-Muller Codes (Quantum)"
  aliases: ["quantum Reed-Muller", "[[15,1,3]] code"]
  see_also: ["transversal-gates", "magic-state-distillation", "triorthogonal-codes"]
- topic_slug: "depolarizing-noise"
  title: "Depolarizing Noise"
  aliases: ["depolarizing channel", "symmetric Pauli noise"]
  see_also: ["pauli-channels", "circuit-level-noise", "biased-noise"]
- topic_slug: "measurement-errors"
  title: "Measurement Errors"
  aliases: ["readout errors", "syndrome measurement noise"]
  see_also: ["syndrome-extraction", "circuit-level-noise", "mwpm-decoder"]
```

### P1 — Modern codes, decoders, missing pieces

```yaml
- topic_slug: "qldpc-codes"
- topic_slug: "bivariate-bicycle-codes"   # IBM BB / gross code
- topic_slug: "floquet-codes"             # Hastings-Haah honeycomb
- topic_slug: "subsystem-codes"
- topic_slug: "concatenated-codes"
- topic_slug: "gkp-codes"
- topic_slug: "cat-codes"
- topic_slug: "biased-noise"
- topic_slug: "triorthogonal-codes"
- topic_slug: "code-deformation"          # code switching, gauge fixing
- topic_slug: "solovay-kitaev"
- topic_slug: "t-gate-count"
- topic_slug: "mwpm-decoder"
- topic_slug: "bp-osd-decoder"
- topic_slug: "union-find-decoder"
- topic_slug: "tensor-network-decoders"
- topic_slug: "neural-decoders"
- topic_slug: "color-code-decoders"
- topic_slug: "circuit-level-noise"       # SD6, SI1000
- topic_slug: "coherent-errors"
- topic_slug: "leakage"
- topic_slug: "pauli-channels"            # + twirling
- topic_slug: "pauli-frame-tracking"
```

### P2 — Theory depth & hardware platforms

```yaml
- topic_slug: "anyons"
- topic_slug: "topological-order"
- topic_slug: "crosstalk"
- topic_slug: "xzzx-surface-code"
- topic_slug: "superconducting-qec"
- topic_slug: "neutral-atom-qec"
- topic_slug: "trapped-ion-qec"
- topic_slug: "photonic-qec"
- topic_slug: "continuous-variable-qec"
```

(Full schema with `description` strings for each page is in `temp/agent_outputs_canonical_topics.yaml` — to be copied into `knowledge/canonical/_roster.yaml` during scaffolding.)

## A.2 Verified arXiv papers (24, ready to ingest as-is)

These 24 papers were verified against the arXiv API. Year may be off by 1 (submission vs publication) but title and authors match.

```yaml
# Fully verified (12)
- {id: "quant-ph/9601029", year: 1996, authors: ["Andrew Steane"], title: "Multiple Particle Interference and Quantum Error Correction", tier: foundational}
- {id: "quant-ph/9705052", year: 1997, authors: ["Daniel Gottesman"], title: "Stabilizer Codes and Quantum Error Correction (PhD thesis)", tier: foundational}
- {id: "quant-ph/9707021", year: 1997, authors: ["A. Yu. Kitaev"], title: "Fault-tolerant quantum computation by anyons", tier: foundational}
- {id: "quant-ph/9906129", year: 1999, authors: ["Dorit Aharonov", "Michael Ben-Or"], title: "Fault-Tolerant Quantum Computation With Constant Error Rate", tier: foundational}
- {id: "quant-ph/9807006", year: 1998, authors: ["Daniel Gottesman"], title: "The Heisenberg Representation of Quantum Computers", tier: foundational}
- {id: "quant-ph/0605138", year: 2006, authors: ["H. Bombin", "M. A. Martin-Delgado"], title: "Topological Quantum Distillation", tier: foundational}
- {id: "1905.06903",       year: 2019, authors: ["Daniel Litinski"], title: "Magic State Distillation: Not as Costly as You Think", tier: peer_reviewed_landmark}
- {id: "2408.13687",       year: 2024, authors: ["Rajeev Acharya", "..."], title: "Quantum error correction below the surface code threshold (Willow)", tier: peer_reviewed_landmark}
- {id: "2310.05900",       year: 2023, authors: ["Johannes Bausch", "..."], title: "Learning to Decode the Surface Code with a Recurrent, Transformer-based Neural Network", tier: peer_reviewed_landmark}
- {id: "2103.02202",       year: 2021, authors: ["Craig Gidney"], title: "Stim: a fast stabilizer circuit simulator", tier: peer_reviewed_landmark}
- {id: "2107.07505",       year: 2021, authors: ["C. Ryan-Anderson", "..."], title: "Realization of real-time fault-tolerant quantum error correction", tier: peer_reviewed_landmark}
- {id: "2409.04628",       year: 2024, authors: ["Ben W. Reichardt", "..."], title: "Demonstration of quantum computation and error correction with a tesseract code", tier: peer_reviewed_landmark}

# Title verified, year off by 1 (also usable — 12)
- {id: "quant-ph/0403025", year: 2004, authors: ["Sergey Bravyi", "Alexei Kitaev"], title: "Universal Quantum Computation with ideal Clifford gates and noisy ancillas", tier: foundational}
- {id: "quant-ph/0110143", year: 2001, authors: ["Eric Dennis", "Alexei Kitaev", "Andrew Landahl", "John Preskill"], title: "Topological quantum memory", tier: foundational}
- {id: "2111.03654",       year: 2021, authors: ["Pavel Panteleev", "Gleb Kalachev"], title: "Asymptotically Good Quantum and Locally Testable Classical LDPC Codes", tier: foundational}
- {id: "1808.02892",       year: 2018, authors: ["Daniel Litinski"], title: "A Game of Surface Codes: Large-Scale Quantum Computing with Lattice Surgery", tier: foundational}
- {id: "2207.06431",       year: 2022, authors: ["Google Quantum AI"], title: "Suppressing quantum errors by scaling a surface code logical qubit", tier: peer_reviewed_landmark}
- {id: "2308.07915",       year: 2023, authors: ["S. Bravyi", "A. W. Cross", "J. M. Gambetta", "D. Maslov", "P. Rall", "T. J. Yoder"], title: "High-threshold and low-overhead fault-tolerant quantum memory", tier: peer_reviewed_landmark}
- {id: "2202.11829",       year: 2022, authors: ["A. Paetznick", "..."], title: "Performance of planar Floquet codes with Majorana-based qubits", tier: recent_significant}
- {id: "2009.07851",       year: 2020, authors: ["J. Pablo Bonilla Ataides", "..."], title: "The XZZX Surface Code", tier: recent_significant}
- {id: "2112.03708",       year: 2021, authors: ["S. Krinner", "..."], title: "Realizing Repeated Quantum Error Correction in a Distance-Three Surface Code", tier: peer_reviewed_landmark}
- {id: "2211.09116",       year: 2022, authors: ["V. V. Sivak", "A. Eickbusch", "..."], title: "Real-time quantum error correction beyond break-even", tier: peer_reviewed_landmark}
- {id: "2208.05461",       year: 2022, authors: ["A. Kubica", "A. Haim", "..."], title: "Erasure qubits: Overcoming the T1 limit in superconducting circuits", tier: recent_significant}
- {id: "2010.02905",       year: 2020, authors: ["J. E. Bourassa", "R. N. Alexander", "M. Vasmer", "..."], title: "Blueprint for a Scalable Photonic Fault-Tolerant Quantum Computer", tier: recent_significant}
```

## A.3 Additional QEC papers surfaced during verification (bonus, ready to ingest)

The verification pass found that 35 of the agent's wrong IDs *coincidentally* point to other QEC-relevant papers — different from what the agent thought, but still excellent corpus additions. Sample of the highest-value ones:

```yaml
- {id: "quant-ph/0703272", year: 2007, authors: ["H. Bombin", "M. A. Martin-Delgado"], title: "Optimal Resources for Topological 2D Stabilizer Codes: Comparative Study"}
- {id: "0803.0272",        year: 2008, authors: ["Austin G. Fowler", "..."], title: "High threshold universal quantum computation on the surface code"}
- {id: "0904.2557",        year: 2009, authors: ["Daniel Gottesman"], title: "An Introduction to Quantum Error Correction and Fault-Tolerant Quantum Computation"}
- {id: "1208.0928",        year: 2012, authors: ["Austin G. Fowler", "Matteo Mariantoni", "John M. Martinis", "Andrew N. Cleland"], title: "Surface codes: Towards practical large-scale quantum computation"}
- {id: "2107.02194",       year: 2021, authors: ["Matthew B. Hastings", "Jeongwan Haah"], title: "Dynamically Generated Logical Qubits"}
- {id: "2202.13641",       year: 2022, authors: ["Anthony Leverrier", "Gilles Zémor"], title: "Quantum Tanner codes"}
- {id: "2305.03828",       year: 2023, authors: ["S. A. Moses", "..."], title: "A Race Track Trapped-Ion Quantum Processor"}
- {id: "1905.00450",       year: 2019, authors: ["Jeremy Guillaud", "Mazyar Mirrahimi"], title: "Bias-preserving gates with stabilized cat qubits"}
- {id: "2312.04522",       year: 2023, authors: ["Craig Gidney", "..."], title: "Yoked surface codes"}
- {id: "2308.08648",       year: 2023, authors: ["Qian Xu", "..."], title: "Constant-Overhead Fault-Tolerant Quantum Computation with Reconfigurable Atom Arrays"}
- {id: "2407.13826",       year: 2024, authors: ["Peter-Jan H. S. Derks", "..."], title: "Designing fault-tolerant circuits using detector error models"}
- {id: "2310.16982",       year: 2023, authors: ["Guanyu Zhu", "..."], title: "Non-Clifford and parallelizable fault-tolerant logical gates on constant-rate quantum Tanner codes via partial gauge fixing"}
- {id: "2309.10331",       year: 2023, authors: ["Alex Fischer", "..."], title: "Hardness results for decoding the surface code with Pauli noise"}
- # … plus ~22 more (full list in temp/arxiv_verification.json)
```

## A.4 Candidate paper list (still requires title-and-author resolution)

For the remaining ~40 papers the agent referenced (Shor 9-qubit, original Steane, Calderbank-Shor, Aharonov-Ben-Or threshold, Bravyi-Kitaev MSD, Fowler-Stephens-Groszkowski, Tillich-Zemor, AlphaQubit, PyMatching v2, etc.) — **the agent's IDs are wrong, but the titles+authors are correct.** The v1 ingestion pipeline's `chatqec ingest --resolve-by-title` (§7.4) is the way to import these without trusting the bad IDs. Resolution targets:

```yaml
- {title_kw: "scheme for reducing decoherence", authors: ["Peter Shor"], expected_year: 1995}
- {title_kw: "good quantum error-correcting codes exist", authors: ["Calderbank", "Shor"], expected_year: 1996}
- {title_kw: "mixed-state entanglement and quantum error correction", authors: ["Bennett", "DiVincenzo", "Smolin", "Wootters"], expected_year: 1996}
- {title_kw: "stabilizer codes saturating quantum hamming", authors: ["Gottesman"], expected_year: 1996}
- {title_kw: "theory of fault-tolerant quantum computation", authors: ["Gottesman"], expected_year: 1998}
- {title_kw: "resilient quantum computation", authors: ["Knill", "Laflamme", "Zurek"], expected_year: 1996}
- {title_kw: "fault-tolerant quantum computation", authors: ["Shor"], expected_year: 1996}
- {title_kw: "quantum accuracy threshold concatenated distance-3", authors: ["Aliferis", "Gottesman", "Preskill"], expected_year: 2006}
- {title_kw: "quantum computing with realistically noisy", authors: ["Knill"], expected_year: 2005}
- {title_kw: "topological computation without braiding", authors: ["Bombin", "Martin-Delgado"], expected_year: 2007}
- {title_kw: "good qLDPC codes with linear time decoders", authors: ["Leverrier", "Zémor"], expected_year: 2022}
- {title_kw: "logical quantum processor reconfigurable atom arrays", authors: ["Bluvstein"], expected_year: 2024}
- {title_kw: "learning high-accuracy error decoding", authors: ["Bausch", "DeepMind"], expected_year: 2024}
- {title_kw: "PyMatching v2", authors: ["Higgott", "Gidney"], expected_year: 2025}
- {title_kw: "dynamically generated logical qubits honeycomb", authors: ["Hastings", "Haah"], expected_year: 2021}
- {title_kw: "fault-tolerant honeycomb memory", authors: ["Gidney", "Newman", "Fowler", "Broughton"], expected_year: 2021}
- {title_kw: "fusion-based quantum computation", authors: ["Bartolucci", "Bombin", "Birchall"], expected_year: 2023}
- {title_kw: "color code on trapped-ion processor", authors: ["Ryan-Anderson", "Quantinuum"], expected_year: 2023}
- # … ~25 more in temp/agent_outputs_papers.yaml (annotated)
```

## A.5 GitHub repos (32, unverified — first ingestion run is the verification pass)

Source: repos research agent (offline). Stars/last_commit dates are estimates; the first `chatqec ingest --github` call validates each repo and corrects metadata.

```yaml
# Simulators
- {repo: "quantumlib/Stim", category: simulator, license: "Apache-2.0"}
- {repo: "Qiskit/qiskit-aer", category: simulator, license: "Apache-2.0"}
- {repo: "quantumlib/Cirq", category: simulator, license: "Apache-2.0"}
- {repo: "rigetti/pyquil", category: simulator, license: "Apache-2.0"}
- {repo: "NVIDIA/cuQuantum", category: simulator, license: "BSD-3-Clause"}
- {repo: "softwareqinc/staq", category: simulator, license: "MIT"}
- {repo: "PennyLaneAI/pennylane", category: simulator, license: "Apache-2.0"}

# Decoders
- {repo: "oscarhiggott/PyMatching", category: decoder, license: "Apache-2.0"}
- {repo: "oscarhiggott/BeliefMatching", category: decoder, license: "Apache-2.0"}
- {repo: "quantumgizmos/ldpc", category: decoder, license: "MIT"}
- {repo: "quantumgizmos/bp_osd", category: decoder, license: "MIT"}
- {repo: "quantumlib/tesseract-decoder", category: decoder, license: "Apache-2.0"}
- {repo: "qecsim/qecsim", category: decoder, license: "BSD-3-Clause"}
# Slugs flagged uncertain by the agent (verify on first ingest):
- {repo: "MITLLRayQC/relay-bp", category: decoder, license: "MIT", verified: false}
- {repo: "inmzhang/strim", category: decoder, license: "MIT", verified: false}

# QEC libraries
- {repo: "qiskit-community/qiskit-qec", category: qec-library, license: "Apache-2.0"}
- {repo: "panqec/panqec", category: qec-library, license: "MIT"}
- {repo: "tqec/tqec", category: qec-library, license: "Apache-2.0", verified: false}
- {repo: "Strilanc/honeycomb_threshold", category: qec-library, license: "Apache-2.0"}
- {repo: "QuantumSavory/QuantumClifford.jl", category: qec-library, license: "MIT"}
- {repo: "PECOS-packages/PECOS", category: qec-library, license: "Apache-2.0"}
- {repo: "microsoft/Quantum", category: qec-library, license: "MIT"}

# Code zoos
- {repo: "errorcorrectionzoo/eczoo_data", category: code-zoo, license: "CC-BY-SA-4.0"}
- {repo: "errorcorrectionzoo/eczoo_generator", category: code-zoo, license: "MIT"}

# Tutorials
- {repo: "qiskit-community/qiskit-textbook", category: tutorial, license: "Apache-2.0 + CC-BY-SA"}
- {repo: "microsoft/QuantumKatas", category: tutorial, license: "MIT"}
- {repo: "oscarhiggott/PyMatching", category: tutorial, license: "Apache-2.0", ingest_path: "docs/examples"}
```

## A.6 YouTube channels (~25, unverified URLs)

Source: videos research agent (offline). Transcript-availability flags reflect the agent's knowledge baseline. First `yt-dlp` call will validate URLs and pull transcripts.

```yaml
- {name: "Google Quantum AI", url: "https://www.youtube.com/@GoogleQuantumAI", focus: high, strategy: "all videos with qec keywords"}
- {name: "Qiskit", url: "https://www.youtube.com/@qiskit", focus: mixed, strategy: "curated subset"}
- {name: "IBM Research", url: "https://www.youtube.com/@IBMResearch", focus: mixed, strategy: "curated subset"}
- {name: "Microsoft Research", url: "https://www.youtube.com/@MicrosoftResearch", focus: mixed, strategy: "curated subset"}
- {name: "Quantinuum", url: "https://www.youtube.com/@quantinuum", focus: high, strategy: "all videos with qec keywords"}
- {name: "Riverlane", url: "https://www.youtube.com/@Riverlane", focus: high, strategy: "all videos"}
- {name: "PsiQuantum", url: "https://www.youtube.com/@PsiQuantum", focus: high, strategy: "all videos"}
- {name: "Caltech IQIM", url: "https://www.youtube.com/@IQIMCaltech", focus: high, strategy: "curated subset"}
- {name: "Simons Institute", url: "https://www.youtube.com/@SimonsInstituteTOC", focus: mixed, strategy: "specific playlists"}
- {name: "Perimeter Institute", url: "https://www.youtube.com/@PerimeterInstitute", focus: mixed, strategy: "curated subset"}
- {name: "KITP", url: "https://www.youtube.com/@KITP_UCSB", focus: mixed, strategy: "curated subset"}
- {name: "IPAM UCLA", url: "https://www.youtube.com/@IPAMUCLA", focus: mixed, strategy: "curated subset"}
- {name: "IQC Waterloo", url: "https://www.youtube.com/@QuantumIQC", focus: mixed, strategy: "curated subset"}
- {name: "MIT OCW", url: "https://www.youtube.com/@mitocw", focus: mixed, strategy: "specific playlists (8.370, 8.371)"}
- {name: "AWS / Amazon Braket", url: "https://www.youtube.com/@AWSDevelopers", focus: mixed, strategy: "curated subset"}
# Note: PIRSA (pirsa.org) is the canonical source for Perimeter content, preferred over YouTube mirrors.
```

## A.7 Textbooks, lecture notes, theses (~22, with licensing analysis)

Source: textbooks research agent. Licensing analysis is human-judged and stable independent of web access.

```yaml
# Commercial — LINK ONLY, never ingest
- {title: "Quantum Computation and Quantum Information (Nielsen & Chuang)", license: commercial, ingest_rec: "do_not_ingest_link_only", note: "Ch 10 ~70 pages on QEC"}
- {title: "Quantum Error Correction (Lidar & Brun, eds.)", license: commercial, ingest_rec: "do_not_ingest_link_only", note: "Cambridge 2013, 666 pages"}
- {title: "Quantum Computer Science (Mermin)", license: commercial, ingest_rec: "do_not_ingest_link_only", note: "Ch 5 ~40 pages on QEC"}

# Free, ingest-recommended
- {title: "Stabilizer Codes and Quantum Error Correction (Gottesman PhD thesis)", id: "quant-ph/9705052", license: free_pdf, ingest_rec: ingest}
- {title: "A Game of Surface Codes (Litinski)", id: "1808.02892", license: free_pdf, ingest_rec: ingest}
- {title: "Surface codes: Towards practical large-scale QC (Fowler et al.)", id: "1208.0928", license: free_pdf, ingest_rec: ingest}
- {title: "Quantum Error Correction for Quantum Memories (Terhal, RMP)", id: "1302.3428", license: free_pdf, ingest_rec: ingest}
- {title: "Quantum Error Correction: An Introductory Guide (Roffe)", id: "1907.11157", license: free_pdf, ingest_rec: ingest}
- {title: "An Introduction to QEC and Fault-Tolerant QC (Gottesman 2009 tutorial)", id: "0904.2557", license: free_pdf, ingest_rec: ingest}
- {title: "Quantum Low-Density Parity-Check Codes (Breuckmann & Eberhardt, PRX Quantum review)", id: "2103.06309", license: free_pdf, ingest_rec: ingest}
- {title: "The XZZX Surface Code", id: "2009.07851", license: free_pdf, ingest_rec: ingest}

# Free lecture notes — ingest with author courtesy notice
- {title: "Lecture Notes for Ph 219/CS 219 (Preskill, Caltech)", url: "http://theory.caltech.edu/~preskill/ph219/", license: free_pdf, ingest_rec: ingest, note: "Ch 7 ~120 pages on QEC, plus Ch 8 on FT"}
- {title: "Surviving as a Quantum Computer in a Classical World (Gottesman 2024 draft textbook)", license: free_pdf, ingest_rec: "ingest_excerpts_under_fair_use", note: "~500+ pages"}
- {title: "Quantum Information and Computation Lecture Notes (Watrous, CS 766/QIC 820)", url: "https://cs.uwaterloo.ca/~watrous/LectureNotes.html", license: free_pdf, ingest_rec: "ingest_excerpts_under_fair_use"}
- {title: "Quantum Error Correction and Fault Tolerance (Steane review)", id: "quant-ph/0304016", license: free_pdf, ingest_rec: ingest}

# Online resources — CC-licensed, ingest fully
- {title: "The Error Correction Zoo", url: "https://errorcorrectionzoo.org", license: "CC-BY-4.0", ingest_rec: ingest, note: "verify license on /about page"}
- {title: "Wikipedia: Quantum Error Correction and related", license: "CC-BY-SA-4.0", ingest_rec: ingest}

# Theory of Quantum Information (Watrous), Wilde Quantum Information Theory — partial QEC content, free PDFs, ingest excerpts under fair use
- {title: "The Theory of Quantum Information (Watrous)", url: "https://cs.uwaterloo.ca/~watrous/TQI/", license: free_pdf, ingest_rec: "ingest_excerpts_under_fair_use"}
- {title: "Quantum Information Theory 2nd ed. (Wilde)", id: "1106.1445", license: free_pdf, ingest_rec: "ingest_excerpts_under_fair_use"}
```

## A.8 Online courses (~26, unverified URLs)

Source: courses research agent. URLs verified on first ingest.

```yaml
# University courses (free)
- {name: "MIT OCW 8.371 QIS II", url: "https://ocw.mit.edu/courses/8-371-quantum-information-science-ii-spring-2018/", ingest_rec: ingest}
- {name: "MIT OCW 8.370 QIS I",  url: "https://ocw.mit.edu/courses/8-370x-quantum-information-science-i-spring-2018/", ingest_rec: ingest}
- {name: "Caltech Ph 219 / CS 219 (Preskill)", url: "http://theory.caltech.edu/~preskill/ph229/", ingest_rec: ingest}
- {name: "Berkeley CS 191 (Vazirani)", url: "https://inst.eecs.berkeley.edu/~cs191/", ingest_rec: ingest}
- {name: "Waterloo CS 798 / CO 681 (Watrous)", url: "https://cs.uwaterloo.ca/~watrous/QIT-notes/", ingest_rec: ingest}
- {name: "NPTEL Quantum Information and Computing", url: "https://nptel.ac.in/courses/115/106/115106066/", ingest_rec: ingest}

# Industry / educational platforms
- {name: "IBM Quantum Learning — QEC course", url: "https://learning.quantum.ibm.com/", ingest_rec: ingest, note: "has dedicated QEC course"}
- {name: "Qiskit Textbook — QEC chapter", url: "https://github.com/Qiskit/textbook", ingest_rec: ingest}
- {name: "PennyLane Codebook", url: "https://pennylane.ai/codebook/", ingest_rec: ingest}
- {name: "Microsoft Quantum Katas — QEC kata", url: "https://github.com/microsoft/QuantumKatas", ingest_rec: ingest}
- {name: "Cirq tutorials (Stabilizer / QEC)", url: "https://quantumai.google/cirq/experiments", ingest_rec: ingest}
- {name: "Stim documentation", url: "https://github.com/quantumlib/Stim", ingest_rec: ingest}

# Summer schools / intensive
- {name: "PSI / PIRSA (Perimeter)", url: "https://pirsa.org/", ingest_rec: ingest, note: "Gottesman QEC series + PSI courses"}
- {name: "Qiskit Global Summer School", url: "https://qiskit.org/events/summer-school/", ingest_rec: ingest}
- {name: "ICTP quantum schools", url: "https://www.ictp.it/events", ingest_rec: ingest}

# Freemium / link-only
- {name: "edX TU Delft (QuTech Academy)", url: "https://www.edx.org/school/delftx", ingest_rec: link_only}
- {name: "Microsoft Learn Quantum", url: "https://learn.microsoft.com/en-us/training/paths/quantum-computing-fundamentals/", ingest_rec: link_only}
- {name: "Q-CTRL Black Opal", url: "https://q-ctrl.com/black-opal", ingest_rec: link_only}
- {name: "Boulder Summer School", url: "https://boulderschool.yale.edu/", ingest_rec: "summarize_in_canonical_wiki"}
- {name: "Les Houches Physics School", url: "https://houches.univ-grenoble-alpes.fr/", ingest_rec: "summarize_in_canonical_wiki"}
```

## A.9 Blogs / RSS (manual seed list — research agent was web-blocked)

The blogs agent could not produce a draft without web access. Below is a maintainer-prepared seed list to be added in v0.5 after human URL verification. Every entry MUST be opened in a browser and the RSS feed discovered before being added to `watchlist.yaml`.

```yaml
# Industry / corporate
- {name: "IBM Research Blog", expected_url: "https://research.ibm.com/blog", verified: false}
- {name: "Google Quantum AI blog", expected_url: "https://quantumai.google/blog", verified: false}
- {name: "AWS Quantum Computing", expected_url: "https://aws.amazon.com/blogs/quantum-computing/", verified: false, note: "may be consolidated post-2024"}
- {name: "Quantinuum blog", expected_url: "https://www.quantinuum.com/blog", verified: false}
- {name: "Riverlane blog", expected_url: "https://www.riverlane.com/news", verified: false}
- {name: "PsiQuantum news", expected_url: "https://psiquantum.com/news/", verified: false}

# Journals (high-quality RSS, must check)
- {name: "Quantum journal", expected_url: "https://quantum-journal.org", expected_rss: "https://quantum-journal.org/feed/", verified: false}
- {name: "PRX Quantum", expected_url: "https://journals.aps.org/prxquantum/", expected_rss: "https://journals.aps.org/prxquantum/rss", verified: false}
- {name: "npj Quantum Information", expected_url: "https://www.nature.com/npjqi/", verified: false}

# Independent
- {name: "Quanta Magazine (quantum tag)", expected_url: "https://www.quantamagazine.org/tag/quantum-computing/", verified: false}
- {name: "Scott Aaronson — Shtetl-Optimized", expected_url: "https://scottaaronson.blog/", expected_rss: "https://scottaaronson.blog/?feed=rss2", verified: false}
- {name: "Quantum Computing Report", expected_url: "https://quantumcomputingreport.com/", verified: false}
- {name: "The Quantum Insider", expected_url: "https://thequantuminsider.com/", verified: false}
```

## A.10 Verification findings: LLM-generated arXiv IDs are unreliable

Verification was performed by `temp/verify_arxiv.py` against the arXiv Atom API on 84 IDs claimed by the papers research agent.

| Result class | Count | % |
|---|---|---|
| Fully correct (title + year match) | 12 | 14% |
| Title correct, year off-by-1 (submission vs publication) | 12 | 14% |
| Wrong ID — points to a different but QEC-relevant paper | 35 | 42% |
| Wrong ID — points to a completely unrelated paper (cosmology, robotics, etc.) | 25 | 30% |

**Implication baked into the design:** every ingestion path for arXiv content (`chatqec ingest --arxiv ID`) calls the arXiv API to verify title+author+year against expectations before writing to the registry. The recommended interface for adding LLM-suggested papers is `chatqec ingest --resolve-by-title "title kw" --author "first author"`, which uses the arXiv search API and is robust to ID hallucination.

Full per-entry verification output: `temp/arxiv_verification.json` (kept for traceability; not part of the published spec).

---

# Appendix B — Known issues to resolve in v0.5

1. Re-run the blogs research with web access to produce a verified RSS list (Appendix A.9 placeholder → actual entries).
2. Resolve the ~40 LLM-suggested papers in Appendix A.4 via the title-and-author search resolver.
3. Verify GitHub repo slugs flagged `verified: false` (Appendix A.5).
4. Verify YouTube channel URLs (Appendix A.6) and probe transcript availability per channel.
5. Verify online course URLs (Appendix A.8); check whether IBM Quantum Learning's QEC course is open-licensed for ingestion or link-only.
6. Confirm Error Correction Zoo licensing (`https://errorcorrectionzoo.org/about`) before bulk ingestion.
7. Contact Preskill for an explicit redistribution waiver on Ph 219 notes (historically generous).
8. Decide policy: are arXiv-license-default papers ingestable for chatbot context (fair use research/educational tool), or do we require CC-licensed papers only? Maintainer call.

---

End of design document.
