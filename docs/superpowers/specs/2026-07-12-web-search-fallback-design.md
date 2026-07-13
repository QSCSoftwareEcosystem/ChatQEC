# Web-Search Fallback (Tavily) — Design

**Date:** 2026-07-12
**Status:** Approved (design)
**Scope:** Add an optional web-search fallback so ChatQEC can answer from the web when
the curated corpus (RAG + canonical wiki) lacks evidence, instead of refusing.

## Motivation

Today `ask()` refuses when retrieved context is insufficient ("I don't have evidence in
the knowledge base for this"). For questions outside the corpus, we want a **fallback**:
search the web, ground the answer in the results, and cite the URLs — while keeping the
curated corpus authoritative (web evidence must never outrank canonical/papers).

## Why Tavily (backend choice)

- **RAG-optimized:** Tavily's `/search` returns already-extracted, clean page **content**
  per result (not just SERP links), which is exactly what a grounded, cited answer needs.
  Serper/SerpAPI return raw Google links that we'd then have to fetch + defuddle ourselves.
- **Simple REST + existing dep:** a single `POST https://api.tavily.com/search` with an API
  key — callable with `requests` (already a project dependency), so **no new SDK**.
- **Relevance scores + URLs** come back per result, mapping directly onto our
  `RetrievedChunk` (`score`, `payload["url"]`).
- **Free tier** suffices for a research assistant's fallback volume.
- Alternatives considered: **Exa** (neural/research search — strong, but overlaps Tavily's
  value and adds a second key), **Serper.dev** (CrewAI's default; raw SERP, needs a fetch
  step), **DuckDuckGo** (`ddgs`, no key but rate-limited/flaky). Tavily wins on
  content-quality-per-integration-effort.

## Architecture

Non-agentic, deterministic **confidence gate** — the pipeline (not the LLM) decides to
search. One new module + small edits, all in the main repo (`chatqec`).

### Components

**1. `src/chatqec/retrieval/web_search.py` (new)**
```
@dataclass(frozen=True)
class WebResult:
    title: str
    url: str
    content: str
    score: float

def web_search(query: str, *, k: int, api_key: str, timeout: float = 15.0) -> list[WebResult]
```
- Lazy-import `requests` (`# noqa: PLC0415`).
- `POST https://api.tavily.com/search` with `{api_key, query, max_results=k,
  search_depth="basic"}`; parse `results[]` → `WebResult`.
- **Never raises.** Empty `api_key` → `[]`. `requests.RequestException` / non-200 /
  malformed JSON → `[]`. This makes the whole feature best-effort.

**2. New `web` authority tier**
- Add `"web": 0.3` to `TIER_WEIGHTS` in `src/chatqec/retrieval/scoring.py` (below
  `blog` = 0.5). Guarantees web evidence ranks last and never outranks corpus content.

**3. Confidence gate in `ChatQEC.ask()` (`src/chatqec/api.py`)**
- After `retrieval = self.retriever.retrieve(...)`, compute a weakness signal:
  `weak = (not retrieval.chunks) or (top rerank score < threshold (0.2))`, reading
  the absolute cross-encoder rerank score from `retrieval.debug["top_rerank_score"]`
  (not the min-max-normalized `c.score`), **and** `not retrieval.matched_topics` (a force-injected
  canonical topic means the corpus already has an authoritative anchor — don't search).
- If `cfg.web_fallback` is true **and** `weak`: call `web_search(question, k=cfg.web_k,
  api_key=...)`; wrap each `WebResult` as
  `RetrievedChunk(chunk_id="web:<sha8>#0", source_id="web:<sha8>",
  text=content, title=title, authority_tier="web", score=score, payload={"url": url})`
  where `<sha8>` = `sha256(url)[:8]`; **append** to the list handed to `synthesize`
  (as part of `retrieved`), so the existing citation/confidence machinery treats them
  uniformly.
- Refactor the ask body minimally: the block from retrieval to `synthesize` gains the
  gated augmentation. `ask_stream()` gets the same augmentation (extract a helper
  `_maybe_web_augment(question, retrieval) -> list[RetrievedChunk]` used by both).

**4. Prompt (`src/chatqec/conversation/prompts.py`)**
- Replace the hard refusal line with: *"For questions beyond the retrieved context: if
  web results (authority tier `web`) are present, you may answer from them — but state
  plainly that the answer draws on web sources rather than the curated knowledge base,
  and cite the source URL. If there is no relevant context at all, refuse politely."*

**5. Provenance / persistence**
- Web chunks carry their URL and are cited like any source. **Ephemeral by default** —
  not written to the registry or Qdrant.
- When the answer used ≥1 web chunk, the CLI (`ask` command) prints a one-line
  suggestion: `tip: to keep a source, run: chatqec ingest --url <url>` for each web URL
  cited — reusing the existing `ingest --url` path for promotion. (Auto-promotion by
  search frequency is a future enhancement, out of scope.)

### Config / keys (`src/chatqec/config.py`, `.env`)
- `.env`: `TAVILY_API_KEY=...` (loaded via existing `load_dotenv`).
- `RetrieverConfig` gains: `web_fallback: bool = False`, `web_fallback_threshold: float
  = 0.45`, `web_k: int = 5`. Feature is **off by default** — zero behavior change unless
  explicitly enabled *and* a key is present.

## Data flow

`ask(question)` → classify → rewrite → `retriever.retrieve` → **[gate: weak & enabled →
`web_search` → wrap as web-tier RetrievedChunks]** → `synthesize(retrieved + web,
canonical)` → Answer (cites corpus + web URLs; confidence from citation count).

## Error handling

Every failure path degrades to current behavior: no key → `web_fallback` effectively
inert (`web_search` returns `[]`); network/HTTP/JSON error → `[]`; gate never fires when
disabled or when retrieval is strong. No exception from the fallback escapes `ask()`.

## Testing

- `web_search`: mock `requests.post` — happy path (parses results), missing key → `[]`,
  non-200 → `[]`, `RequestException` → `[]`, malformed JSON → `[]`.
- Gate: with a stubbed searcher, weak retrieval (empty / low-score, no matched topics)
  → searcher called and web chunks appended; strong retrieval (high score or matched
  canonical topic) → searcher NOT called. Disabled config → never called.
- Tier: `TIER_WEIGHTS["web"] < TIER_WEIGHTS["blog"]`, and a scoring test that a web chunk
  with equal similarity ranks below a blog chunk.
- Prompt: `SYNTHESIZER_SYSTEM_PROMPT` contains the web-fallback allowance wording.
- No network in tests — all external calls mocked/stubbed.

## Out of scope (YAGNI)

Agentic/LLM-driven search (Claude choosing when to search); multi-step query
reformulation; auto-promotion by search frequency; a `web_search` MCP tool; caching web
results. The searcher is a plain function; upgrading to an agent tool later reuses it.

## Decisions locked

1. Backend: **Tavily** (REST via `requests`, no new SDK).
2. Trigger: **deterministic confidence gate** (non-agentic), off by default.
3. Persistence: **ephemeral** + printed `ingest --url` promotion tip.
4. New **`web` tier = 0.3**, below `blog`.
