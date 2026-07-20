# Web-Search Fallback (Tavily) Implementation Plan

**Goal:** Add an optional, deterministic web-search fallback (Tavily) so `ask()` answers from the web — cited, lowest-authority — when the curated corpus lacks evidence, instead of refusing.

**Architecture:** One new module `retrieval/web_search.py` holding a Tavily REST searcher and a **pure gate function**; a new lowest `web` authority tier; `ChatQEC` gains web params and calls the gate between retrieval and synthesis; a prompt tweak; CLI wiring + docs. Off by default (needs `web_fallback=True` **and** `TAVILY_API_KEY`).

**Tech Stack:** Python 3.11, `requests` (already a dep — no new SDK), pydantic/dataclasses, pytest with mocked HTTP.

## Global Constraints
- Main repo `chatqec` at `~/chatQEC`; use `uv run`. Conventional Commits on `main`, no `Co-Authored-By`.
- **No new dependency** — call Tavily via `requests` (lazy-import inside the function; the repo per-file-ignores `PLC0415` for lazy imports, but `requests` is imported lazily elsewhere without noqa via top-level too — follow the file's own convention, default to a top-level `import requests` since `requests` is a hard dep).
- **`web_search` and the gate never raise** — any failure (missing key, network, non-200, bad JSON) returns `[]`.
- New tier weight: `TIER_WEIGHTS["web"] = 0.3` (strictly below `blog` = 0.5).
- Feature **off by default**: `web_fallback=False`. When off, or when `web_searcher is None`, behavior is byte-identical to today.
- Gate trigger (deterministic): search only when `web_fallback` is on AND retrieval is weak — `(no chunks) or (top rerank score < threshold (0.4))` — AND `not retrieval.matched_topics` (a force-injected canonical topic means the corpus already has an authoritative anchor). The gate reads the absolute cross-encoder rerank score (`debug["top_rerank_score"]`), not the min-max-normalized chunk score.
- Web results are ephemeral (not written to registry/Qdrant). Cite the URL.
- Ruff/mypy/pre-commit must pass (repo runs ruff, ruff-format, mypy on commit).
- Tests must not hit the network — mock `requests.post` / stub the searcher.

## Existing interfaces (verified, for reference)
- `RetrievedChunk(chunk_id: str, source_id: str, text: str, title: str, authority_tier: str, score: float, payload: dict)` — `src/chatqec/retrieval/hybrid.py`.
- `RetrievalResult(chunks: list[RetrievedChunk], matched_topics: list[str], ...)` — same file.
- `Citation(source_id, chunk_id, title, url)` — `src/chatqec/api_types.py`.
- `Synthesizer.synthesize(*, question, retrieved, canonical_chunks, history, images=None)`.
- `sha256_text(s)` — `from chatqec.ingestion.hashing import sha256_text`.
- `TIER_WEIGHTS` — `src/chatqec/retrieval/scoring.py` (canonical 1.0, peer_reviewed 0.9, preprint 0.7, blog 0.5, video 0.5).

---

### Task 1: Tavily searcher — `web_search.py`

**Files:**
- Create: `src/chatqec/retrieval/web_search.py`
- Test: `tests/test_web_search.py`

**Interfaces:**
- Produces: `WebResult(title: str, url: str, content: str, score: float)` (frozen dataclass) and `web_search(query: str, *, k: int, api_key: str, timeout: float = 15.0) -> list[WebResult]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_web_search.py`:
```python
"""Tavily web_search — all HTTP mocked; never hits the network."""

from unittest.mock import MagicMock, patch

from chatqec.retrieval.web_search import WebResult, web_search

_FAKE = {
    "results": [
        {"title": "Surface codes", "url": "https://ex.com/a", "content": "text A", "score": 0.9},
        {"title": "qLDPC", "url": "https://ex.com/b", "content": "text B", "score": 0.7},
    ]
}


def test_web_search_parses_results():
    resp = MagicMock(status_code=200)
    resp.json.return_value = _FAKE
    with patch("requests.post", return_value=resp) as post:
        out = web_search("surface code", k=2, api_key="tvly-x")
    assert [r.url for r in out] == ["https://ex.com/a", "https://ex.com/b"]
    assert out[0] == WebResult(title="Surface codes", url="https://ex.com/a", content="text A", score=0.9)
    # api_key + query + max_results were sent
    _, kwargs = post.call_args
    assert kwargs["json"]["max_results"] == 2


def test_web_search_empty_key_returns_empty_without_calling():
    with patch("requests.post") as post:
        assert web_search("q", k=3, api_key="") == []
    post.assert_not_called()


def test_web_search_non_200_returns_empty():
    resp = MagicMock(status_code=401)
    resp.json.return_value = {}
    with patch("requests.post", return_value=resp):
        assert web_search("q", k=3, api_key="tvly-x") == []


def test_web_search_request_exception_returns_empty():
    import requests

    with patch("requests.post", side_effect=requests.RequestException("boom")):
        assert web_search("q", k=3, api_key="tvly-x") == []


def test_web_search_malformed_json_returns_empty():
    resp = MagicMock(status_code=200)
    resp.json.side_effect = ValueError("bad json")
    with patch("requests.post", return_value=resp):
        assert web_search("q", k=3, api_key="tvly-x") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_web_search.py -v`
Expected: FAIL — module `chatqec.retrieval.web_search` not found.

- [ ] **Step 3: Write the implementation**

Create `src/chatqec/retrieval/web_search.py`:
```python
"""Tavily-backed web search — the corpus fallback source.

A thin, dependency-light client for Tavily's /search endpoint (called with the
already-present `requests`). The chosen backend returns clean extracted page
*content* per result (not just SERP links), so answers can be grounded and cited
without a separate fetch step. The function is best-effort: any failure — missing
key, network error, non-200, malformed body — returns an empty list so the caller
degrades to corpus-only behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

_TAVILY_URL = "https://api.tavily.com/search"


@dataclass(frozen=True)
class WebResult:
    title: str
    url: str
    content: str
    score: float


def web_search(query: str, *, k: int, api_key: str, timeout: float = 15.0) -> list[WebResult]:
    """Search the web via Tavily; return up to ``k`` results. Never raises."""
    if not api_key:
        return []
    try:
        resp = requests.post(
            _TAVILY_URL,
            json={
                "api_key": api_key,
                "query": query,
                "max_results": k,
                "search_depth": "basic",
            },
            timeout=timeout,
        )
        if resp.status_code != 200:
            return []
        payload = resp.json()
    except (requests.RequestException, ValueError):
        return []

    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        return []
    out: list[WebResult] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        out.append(
            WebResult(
                title=str(r.get("title", "")),
                url=str(r.get("url", "")),
                content=str(r.get("content", "")),
                score=float(r.get("score", 0.0) or 0.0),
            )
        )
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_web_search.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**
```bash
cd ~/chatQEC
git add src/chatqec/retrieval/web_search.py tests/test_web_search.py
git commit -m "feat(retrieval): Tavily web_search client (best-effort, no new deps)"
```

---

### Task 2: `web` tier + gate function + `ChatQEC` wiring

**Files:**
- Modify: `src/chatqec/retrieval/scoring.py` (add `"web": 0.3` to `TIER_WEIGHTS`)
- Modify: `src/chatqec/retrieval/web_search.py` (add `build_web_chunks`)
- Modify: `src/chatqec/api.py` (`ChatQEC.__init__` params; call gate in `ask` + `ask_stream`)
- Test: `tests/test_web_search.py` (add gate tests), `tests/test_scoring.py` (tier test)

**Interfaces:**
- Consumes: `WebResult`, `web_search` (Task 1); `RetrievedChunk`, `RetrievalResult`, `sha256_text`.
- Produces: `build_web_chunks(question: str, *, retrieval: RetrievalResult, web_searcher: Callable[[str, int], list[WebResult]] | None, threshold: float, web_k: int) -> list[RetrievedChunk]`; `ChatQEC.__init__` gains `web_searcher=None`, `web_fallback=False`, `web_fallback_threshold=0.45`, `web_k=5`.

- [ ] **Step 1: Add the `web` tier + test**

In `src/chatqec/retrieval/scoring.py`, add to `TIER_WEIGHTS`:
```python
    "web": 0.3,
```
Add to `tests/test_scoring.py`:
```python
def test_web_tier_ranks_below_blog():
    from chatqec.retrieval.scoring import TIER_WEIGHTS

    assert TIER_WEIGHTS["web"] < TIER_WEIGHTS["blog"]
```
Run: `uv run pytest tests/test_scoring.py::test_web_tier_ranks_below_blog -v` → PASS.

- [ ] **Step 2: Write failing gate tests**

Add to `tests/test_web_search.py`:
```python
from chatqec.retrieval.hybrid import RetrievalResult, RetrievedChunk
from chatqec.retrieval.web_search import build_web_chunks


def _chunk(score):
    return RetrievedChunk(chunk_id="c#0", source_id="c", text="t", title="T",
                          authority_tier="preprint", score=score, payload={})


def _stub_searcher(_q, _k):
    return [WebResult(title="W", url="https://ex.com/x", content="web text", score=0.5)]


def _result(chunks, topics):
    # RetrievalResult(chunks, matched_topics, debug={}) — debug defaults, verified.
    return RetrievalResult(chunks=chunks, matched_topics=topics)


def test_gate_searches_when_retrieval_is_weak():
    out = build_web_chunks("q", retrieval=_result([_chunk(0.2)], []),
                           web_searcher=_stub_searcher, threshold=0.45, web_k=5)
    assert len(out) == 1
    assert out[0].authority_tier == "web"
    assert out[0].source_id.startswith("web:")
    assert out[0].payload["url"] == "https://ex.com/x"


def test_gate_searches_when_no_chunks():
    out = build_web_chunks("q", retrieval=_result([], []),
                           web_searcher=_stub_searcher, threshold=0.45, web_k=5)
    assert len(out) == 1


def test_gate_skips_when_retrieval_is_strong():
    out = build_web_chunks("q", retrieval=_result([_chunk(0.8)], []),
                           web_searcher=_stub_searcher, threshold=0.45, web_k=5)
    assert out == []


def test_gate_skips_when_canonical_topic_matched():
    out = build_web_chunks("q", retrieval=_result([_chunk(0.1)], ["surface-code"]),
                           web_searcher=_stub_searcher, threshold=0.45, web_k=5)
    assert out == []


def test_gate_skips_when_searcher_is_none():
    out = build_web_chunks("q", retrieval=_result([], []),
                           web_searcher=None, threshold=0.45, web_k=5)
    assert out == []
```
(`RetrievalResult(chunks, matched_topics, debug={})` — `debug` defaults, so the two-arg construction above is valid.)

Run: `uv run pytest tests/test_web_search.py -k gate -v` → FAIL (`build_web_chunks` undefined).

- [ ] **Step 3: Implement `build_web_chunks`**

Add to `src/chatqec/retrieval/web_search.py`:
```python
from collections.abc import Callable  # add to imports at top

from chatqec.ingestion.hashing import sha256_text  # add to imports at top
from chatqec.retrieval.hybrid import RetrievalResult, RetrievedChunk  # add to imports at top


def build_web_chunks(
    question: str,
    *,
    retrieval: RetrievalResult,
    web_searcher: Callable[[str, int], list[WebResult]] | None,
    threshold: float,
    web_k: int,
) -> list[RetrievedChunk]:
    """Return web-tier chunks when the corpus retrieval is weak, else ``[]``.

    Deterministic gate: search only when a searcher is configured, no canonical
    topic was force-injected, and the retrieval is weak (no chunks, or the best
    chunk score is below ``threshold``). Never raises.
    """
    if web_searcher is None:
        return []
    if retrieval.matched_topics:
        return []
    top = max((c.score for c in retrieval.chunks), default=0.0)
    if retrieval.chunks and top >= threshold:
        return []
    try:
        results = web_searcher(question, web_k)
    except Exception:  # searcher is best-effort; never break the answer path
        return []
    chunks: list[RetrievedChunk] = []
    for r in results:
        sid = f"web:{sha256_text(r.url)[:8]}"
        chunks.append(
            RetrievedChunk(
                chunk_id=f"{sid}#0",
                source_id=sid,
                text=r.content,
                title=r.title,
                authority_tier="web",
                score=r.score,
                payload={"url": r.url},
            )
        )
    return chunks
```
Run: `uv run pytest tests/test_web_search.py -v` → PASS (all, incl. gate tests).

- [ ] **Step 4: Wire into `ChatQEC`**

In `src/chatqec/api.py` `ChatQEC.__init__`, add params after `candidate_k`:
```python
        web_searcher: "Callable[[str, int], list[WebResult]] | None" = None,
        web_fallback: bool = False,
        web_fallback_threshold: float = 0.45,
        web_k: int = 5,
```
and store them:
```python
        self.web_searcher = web_searcher
        self.web_fallback = web_fallback
        self.web_fallback_threshold = web_fallback_threshold
        self.web_k = web_k
```
Add imports at the top of `api.py`:
```python
from collections.abc import Callable

from chatqec.retrieval.web_search import WebResult, build_web_chunks
```
Add a private helper on `ChatQEC`:
```python
    def _web_chunks(self, question: str, retrieval: "RetrievalResult") -> list[RetrievedChunk]:
        searcher = self.web_searcher if self.web_fallback else None
        return build_web_chunks(
            question,
            retrieval=retrieval,
            web_searcher=searcher,
            threshold=self.web_fallback_threshold,
            web_k=self.web_k,
        )
```
In `ask()`, right after `canonical_chunks = self._canonical_chunks(...)` and its trace, add:
```python
        web_chunks = self._web_chunks(question, retrieval)
```
and change the `synthesize(...)` call's `retrieved=` argument from `retrieval.chunks` to:
```python
            retrieved=retrieval.chunks + web_chunks,
```
In `ask_stream()`, make the identical change: after `canonical_chunks = ...`, add `web_chunks = self._web_chunks(question, retrieval)`, and pass `retrieved=retrieval.chunks + web_chunks` to the `synthesize_stream(...)` (or `synthesize(...)`) call.

(If `RetrievedChunk`/`RetrievalResult` aren't already imported in `api.py`, add them from `chatqec.retrieval.hybrid`.)

- [ ] **Step 5: Verify nothing regressed**

Run: `uv run pytest tests/test_web_search.py tests/test_scoring.py tests/test_api_chatqec.py -q`
Expected: PASS (existing api tests still green — the default `web_fallback=False` path is unchanged).
Also `uv run ruff check src/chatqec/retrieval/web_search.py src/chatqec/api.py src/chatqec/retrieval/scoring.py` and `uv run mypy src/chatqec/retrieval/web_search.py` → clean.

- [ ] **Step 6: Commit**
```bash
cd ~/chatQEC
git add src/chatqec/retrieval/web_search.py src/chatqec/retrieval/scoring.py src/chatqec/api.py tests/test_web_search.py tests/test_scoring.py
git commit -m "feat(retrieval): web-tier + deterministic web-search gate wired into ask()"
```

---

### Task 3: Prompt — allow web-fallback instead of hard refusal

**Files:**
- Modify: `src/chatqec/conversation/prompts.py`
- Test: `tests/test_prompts_web.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_prompts_web.py`:
```python
from chatqec.conversation.prompts import SYNTHESIZER_SYSTEM_PROMPT


def test_prompt_allows_web_fallback_with_disclosure():
    p = SYNTHESIZER_SYSTEM_PROMPT.lower()
    assert "web" in p
    # must instruct to disclose web reliance and still refuse when nothing is found
    assert "refuse" in p
```
Run: `uv run pytest tests/test_prompts_web.py -v` → FAIL (no "web" wording yet).

- [ ] **Step 2: Update the prompt**

In `src/chatqec/conversation/prompts.py`, replace the line:
```
- For questions beyond the retrieved context, refuse politely: "I don't have evidence in the knowledge base for this."
```
with:
```
- For questions beyond the retrieved context: if web results (authority tier `web`)
  are present, you MAY answer from them, but state plainly that the answer draws on
  web sources rather than the curated knowledge base, and cite the source URL. If
  there is no relevant context at all, refuse politely: "I don't have evidence in the
  knowledge base for this."
```

- [ ] **Step 3: Run test to verify it passes**

Run: `uv run pytest tests/test_prompts_web.py -v` → PASS.

- [ ] **Step 4: Commit**
```bash
cd ~/chatQEC
git add src/chatqec/conversation/prompts.py tests/test_prompts_web.py
git commit -m "feat(prompt): allow cited web-source answers when corpus lacks evidence"
```

---

### Task 4: CLI wiring + promotion tip + "Why Tavily" doc

**Files:**
- Modify: `src/chatqec/cli.py` (`_build_chatqec`, and the `ask` command's output)
- Modify: `docs/EXAMPLES.md` (add "Why Tavily" rationale)

**Interfaces:**
- Consumes: `web_search` (Task 1), `ChatQEC` web params (Task 2). `Citation.url` for the tip.

- [ ] **Step 1: Wire the searcher in `_build_chatqec`**

In `src/chatqec/cli.py`, inside `_build_chatqec(...)`, before constructing `ChatQEC(...)`, add:
```python
    import os  # if not already imported at top

    from chatqec.retrieval.web_search import web_search

    _tavily_key = os.environ.get("TAVILY_API_KEY", "")
    _web_enabled = os.environ.get("CHATQEC_WEB_FALLBACK", "").lower() in {"1", "true", "yes"}

    def _web_searcher(query: str, k: int):
        return web_search(query, k=k, api_key=_tavily_key)
```
and pass to the `ChatQEC(...)` constructor:
```python
        web_searcher=_web_searcher if _tavily_key else None,
        web_fallback=_web_enabled,
```
(Feature stays off unless `CHATQEC_WEB_FALLBACK` is truthy AND `TAVILY_API_KEY` is set.)

- [ ] **Step 2: Print the promotion tip in the `ask` command**

In `src/chatqec/cli.py`, in the `ask` command after the answer is printed, add:
```python
    web_urls = [c.url for c in answer.citations if c.source_id.startswith("web:") and c.url]
    for u in dict.fromkeys(web_urls):  # de-dup, preserve order
        click.echo(f"tip: to keep this web source, run: chatqec ingest --url {u}", err=True)
```
(Find the exact spot where `answer` is available and citations are already printed; place this just after. If the `ask` command variable is not named `answer`, use the actual name.)

- [ ] **Step 3: Manual smoke (no automated test — CLI is `# pragma: no cover`)**

With no key set, confirm behavior is unchanged:
```bash
cd ~/chatQEC
uv run chatqec ask "What is the surface code threshold?" 2>/dev/null | head -5
```
Expected: a normal cited answer (web fallback inert — no key/flag).

- [ ] **Step 4: Add the "Why Tavily" rationale to `docs/EXAMPLES.md`**

Add a subsection (place it near the QEC-tools / corpus stats area):
```markdown
### Web-search fallback (Tavily) — why this backend

When a question isn't covered by the corpus or canonical wiki, ChatQEC can fall back
to a **web search** (off by default; enable with `CHATQEC_WEB_FALLBACK=true` +
`TAVILY_API_KEY`). The backend is **Tavily**, chosen deliberately:

- **RAG-optimized output** — Tavily returns already-extracted, clean page *content* per
  result, not just SERP links, so answers are grounded and citable without a separate
  fetch/parse step (Serper/SerpAPI would require that extra step).
- **No new dependency** — a single `POST https://api.tavily.com/search` via the
  already-present `requests`; no extra SDK.
- **Per-result relevance scores + URLs** map directly onto our retrieval model and
  citations.
- **Free tier** covers fallback volume for a research assistant.
- Alternatives weighed: **Exa** (neural/research search — strong but a second key with
  overlapping value), **Serper.dev** (CrewAI's default; raw SERP + fetch step),
  **DuckDuckGo** (`ddgs`, no key but rate-limited/flaky). Tavily won on
  content-quality per unit of integration effort.

Web evidence is the **lowest** authority tier (`web` = 0.3, below `blog` = 0.5), so it
can never outrank canonical pages or papers; the model is instructed to disclose when an
answer relies on web sources and to cite the URL. Results are ephemeral — the CLI prints
a `chatqec ingest --url <url>` tip so a repeatedly-useful source can be promoted into the
permanent corpus.
```

- [ ] **Step 5: Commit**
```bash
cd ~/chatQEC
git add src/chatqec/cli.py docs/EXAMPLES.md
git commit -m "feat(cli): wire Tavily web fallback + promotion tip; docs: why-Tavily rationale"
```

---

## Notes for the implementer
- The whole feature is inert unless `web_fallback=True` and a searcher is provided — every existing test path (default construction) is unchanged.
- `web_search` and `build_web_chunks` must never raise; the gate tests and the missing-key/error tests lock this in.
- Do not add a `tavily-python` dependency — `requests` + REST only.
