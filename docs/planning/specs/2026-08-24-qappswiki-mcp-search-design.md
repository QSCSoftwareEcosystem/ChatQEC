# QAppsWiki as a Live Knowledge Source (MCP `search_pages`) — Design

**Date:** 2026-08-24
**Status:** Approved (design)
**Scope:** Let ChatQEC ground answers in the QAppsWiki corpus (631 Error Correction Zoo
codes plus curated QEC/QEM pages) by adding full-text search to QAppsWiki's own MCP server
and calling it during retrieval, alongside Qdrant. No second Qdrant collection, no
ingestion pipeline, no sync job.

**Repos touched:** `QSCSoftwareEcosystem/QAppsWiki` (Phase 1), `chatQEC` (Phase 2).

## Motivation

`concepts/qec/` holds 649 pages (2.4 MB) of QEC knowledge, including the full Error
Correction Zoo import. That is directly on-topic for ChatQEC and larger than the 60-page
canonical wiki it ships with. Meanwhile `code_params` knows only three hardcoded code
families. Getting that content in front of the synthesizer is the goal; doing it without
inheriting a second copy of the corpus is the constraint.

## Why not the obvious alternatives

**Why not the GitHub REST API at query time.** GitHub has not indexed the repo for code
search. Measured 2026-08-24 with an authenticated token:

```
"stabilizer" in QSCSoftwareEcosystem/QAppsWiki -> total_count: 0, incomplete_results: true
"stabilizer" in quantumlib/Stim (control)      -> total_count: 188
```

Code search is also capped at 10 requests/minute. The API can fetch a file by path, but it
cannot find one by meaning, so it cannot serve "which codes tolerate biased noise". A
GitHub MCP server inherits exactly this limitation, because its search tool wraps the same
endpoint.

**Why not ingest into Qdrant.** Workable, and was the leading candidate until edit access
to QAppsWiki was confirmed. It costs a mirror, a HEAD-SHA freshness check, per-page
`content_sha256` tracking, delete-before-reinsert on every update, registry rows, and a
second copy of content that already lives in a maintained repo. Every one of those is a
staleness bug waiting to happen.

**Why not QAppsWiki's existing MCP server as-is.** It already exposes an MCP stdio server
(`tools/qappswiki/serve.py`, "Stage 8"), but its queries run over `graph.json` and return
node metadata and edges. `q_query` substring-matches on id, title, and domains only. It can
say the Steane node exists and who cites it; it cannot return the sentence containing
⟦7^m,1,3^m⟧. Grounding needs prose.

**Chosen approach.** Add prose search to the server that already sits next to the content.
The wiki owns its own retrieval, ChatQEC owns ranking, and there is exactly one copy of the
corpus. Freshness is free because the server reads the working tree.

## Architecture

Deterministic, not agentic. The retriever calls the wiki on every query and merges the
results into the candidate pool; the cross-encoder decides what survives. This mirrors
`web_search.py` in transport shape, but differs in policy: web search is gated behind weak
retrieval, whereas the wiki always competes.

```
question
   |
   +--> HybridRetriever -> Qdrant (dense + BM25)  --+
   |                                                +--> cross-encoder rerank -> top_k -> synthesizer
   +--> MCP: qappswiki.search_pages(query, k) ------+
```

### Phase 1 — QAppsWiki (`tools/qappswiki/`)

**1. `search.py` (new)**

```python
@dataclass(frozen=True)
class Section:
    path: str          # concepts/qec/steane.md
    title: str         # frontmatter `name`
    heading: str       # "## Protection"
    text: str
    provenance_status: str
    sources: list[str]

def build_index(root: Path) -> Index      # section split + BM25 statistics
def search(index: Index, query: str, k: int, domain: str | None) -> list[Hit]
```

- Split each page on `##` headings, keeping frontmatter with every section. Sections, not
  whole files, because the largest page in the repo is 380 KB and would swamp a context
  window.
- Pure-Python BM25 (Okapi, k1=1.5, b=0.75). No new dependency; the package stays on
  `pyyaml` + `networkx`.
- Cache the index to `.qappswiki/search-index.json`, keyed by per-file SHA-256. Rebuild
  only changed files, so first call after an edit is fast and correctness never depends on
  remembering to re-index.
- Exclude `raw/md/` by default. Those are arXiv full texts that ChatQEC already ingests via
  `arxiv_source.py`, and two copies of one paper would compete in the reranker.

**2. `serve.py` — new MCP tool**

```
search_pages(query: str, k: int = 8, domain: str | None = None)
  -> [{path, title, heading, text, score, provenance_status, sources, url}]
```

- `url` is the canonical GitHub blob URL, so ChatQEC can cite a clickable source.
- Registered alongside the existing `q_*` tools; no change to their behavior.

**3. Tests** in the existing `tools/tests/` style: section splitting, BM25 ranking order,
incremental cache invalidation, `raw/md` exclusion, and a real-corpus smoke test matching
the existing `test_real_corpus.py`.

### Phase 2 — ChatQEC

**4. `src/chatqec/conversation/multi_mcp.py` (new)**

ChatQEC today builds exactly one MCP client (`cli.py:258`), so connecting the wiki would
cost the Stim tools. `MCPAdapter` depends only on the `MCPClient` Protocol
(`list_tools()` / `call_tool()`), so a fan-out client satisfying that Protocol needs zero
changes to the adapter, synthesizer, or tool loop.

```python
class MultiMCPClient:
    def __init__(self, clients: Sequence[MCPClient]) -> None: ...
    def list_tools(self) -> list[MCPToolDef]:  # union, first server wins on name clash
    def call_tool(self, name, args) -> dict:   # routed by tool name
```

- A server that fails to start is logged and skipped; the rest keep working.

**5. Config**

```yaml
mcp:
  servers:                      # new; server_command still honored for back-compat
    - "chatqec-mcp-tools"
    - "qappswiki-mcp"
wiki:
  enabled: true
  k: 8
  timeout_seconds: 5
```

**6. `src/chatqec/retrieval/wiki_search.py` (new)**

Modeled on `web_search.py`: call `search_pages`, wrap each hit as a `RetrievedChunk`, and
**never raise**. Timeout, missing server, malformed payload all degrade to `[]` so a wiki
outage cannot break the answer path.

**7. Merge point (`src/chatqec/api.py`)**

Wiki hits join `retrieval.chunks` before reranking, so the cross-encoder ranks wiki prose
against corpus prose on one scale. This is the "whichever gives better context wins"
behavior, decided by the reranker rather than a hand-tuned rule.

**8. Two new authority tiers (`src/chatqec/retrieval/scoring.py`)**

```python
"wiki_curated": 0.6,        # provenance_status: verified, or hand-written pages
"wiki_unverified": 0.35,    # provenance_status: needs-verification (the Zoo import)
```

Rationale: 631 Zoo pages carry `provenance_status: needs-verification` and
`status: provisional` in their own frontmatter. The wiki grades its own trustworthiness, so
ChatQEC should honor that grading rather than flatten it. Unverified imports land between
`blog` (0.5) and `web` (0.3), so they can inform an answer but never outrank a
peer-reviewed paper.

## Attribution and licensing

Zoo-imported pages state: *"Text is reused under CC-BY-SA; attribute the Error Correction
Zoo (errorcorrectionzoo.org), CC-BY-SA."* `search_pages` returns `sources`, and citations
rendered from wiki chunks must carry that attribution. The QAppsWiki tooling is MIT
(`tools/pyproject.toml`); the repo has no top-level LICENSE, which should be raised with
its owners but does not block this work, since CC-BY-SA attribution is satisfied per page.

## Error handling

| Failure | Behavior |
|---|---|
| Wiki MCP server not configured | Feature off; corpus-only answers, no error |
| Server fails to start | Logged once, skipped; Stim tools unaffected |
| `search_pages` times out or errors | Returns `[]`; answer proceeds from Qdrant alone |
| Index cache corrupt | Rebuilt from scratch on next call |
| Qdrant down | Unchanged behavior; wiki does not mask a broken store |

Every wiki failure is non-fatal. The corpus path must never depend on the wiki being
healthy.

## Testing

- **QAppsWiki:** unit tests for splitting, ranking, cache invalidation, exclusions.
- **ChatQEC:** `MultiMCPClient` fan-out and name-collision routing with fakes; `wiki_search`
  fail-open on timeout, bad payload, and missing tool; a merge test asserting wiki chunks
  reach the reranker; tier-weight ordering asserted directly.
- **End-to-end:** a real query against a checked-out wiki, asserting a wiki-sourced citation
  appears for a Zoo-only topic (a code absent from the ChatQEC corpus).
- **Eval:** run the 50-pair set before and after. The wiki must not regress existing
  answers; watch for Zoo pages crowding out papers on questions the corpus already answers.

## Known limitations

- **BM25 is lexical.** "biased noise" will not match a page that says "asymmetric Pauli
  channel". The reranker rescores what BM25 returns but cannot rescue what was never
  retrieved. If recall proves weak, add a local embedding index inside QAppsWiki, which
  keeps the ownership boundary intact.
- **Local checkout required.** The MCP server is a stdio subprocess reading local files, not
  a hosted endpoint. Deployments need the wiki cloned.
- **No near-duplicate suppression** exists in retrieval today. If wiki and corpus chunks say
  the same thing, both can occupy the top_k. Deferred until observed.

## Out of scope

Ingesting `raw/md`; the graph traversal tools (`q_get_neighbors`, `q_shortest_path`) as
LLM-facing tools; the second wiki (`qrag`), which the current token cannot see; wiring
`code_params` to Zoo data, which is a natural follow-on once this proves useful.
