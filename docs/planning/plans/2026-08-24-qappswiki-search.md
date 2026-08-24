# QAppsWiki Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let ChatQEC ground answers in QAppsWiki's 691 QEC/QEM concept pages by adding BM25
full-text search to QAppsWiki's MCP server and calling it during ChatQEC retrieval, so wiki
passages and corpus passages are reranked together.

**Architecture:** QAppsWiki gains a section-level BM25 index and a `search_pages` MCP tool
reading its own working tree, so it stays live with no sync. ChatQEC gains a multi-server MCP
client, calls `search_pages` during retrieval, and feeds the hits into the same cross-encoder
rerank pool as Qdrant results. No second Qdrant collection, no ingestion.

**Tech Stack:** Python 3.10+, pure-stdlib BM25, `mcp>=1.0` stdio, pytest. ChatQEC side:
Qdrant, `mxbai-rerank-base-v1` cross-encoder, Hydra config.

**Spec:** `docs/planning/specs/2026-08-24-qappswiki-mcp-search-design.md`

## Global Constraints

- **Two repos.** Phase 1 in `/home/pza/QAppsWiki` on branch `feature/mcp-search-pages`.
  Phase 2 in `/home/pza/chatQEC` on `main` (no branch; user preference).
- **No new QAppsWiki dependencies.** The package stays on `pyyaml` + `networkx`; `mcp` is an
  optional extra. BM25 is hand-written, no `rank_bm25`.
- **Reuse, don't reinvent.** `collect.collect_pages()` enumerates the corpus and already
  excludes `raw/md/` via `paths.is_ignored()`. `parse.split_frontmatter()` splits frontmatter
  (it returns the raw YAML **string**, not a dict). `cache.Cache` is the content-addressed
  per-page cache.
- **Cache namespace.** `Cache(out_dir)` writes `<out_dir>/cache/<sha256>.json` keyed only by
  file digest, so the search cache MUST use a separate root (`out / "search"`) or it will
  collide with the parse cache for the same file.
- **Every wiki failure is non-fatal in ChatQEC.** Timeout, missing server, malformed payload
  all degrade to `[]` and the answer proceeds from Qdrant alone.
- **Tier weights:** add `"wiki_curated": 0.6` and `"wiki_unverified": 0.35`. Today 689 of 691
  pages are `needs-verification`, so expect nearly everything at 0.35.
- Python: 4-space indent, type hints on public functions, `from __future__ import annotations`.
- ChatQEC commits use Conventional Commits. No `Co-Authored-By` trailer.

## File Structure

**Phase 1 — `/home/pza/QAppsWiki`**

| File | Responsibility |
|---|---|
| `tools/qappswiki/search.py` (new) | Section splitting, BM25 index, corpus build. No MCP, no I/O beyond reading pages. |
| `tools/qappswiki/serve.py` (modify) | Add `q_search_pages` query fn; register `search_pages`; `start_server` gains `content_root`. |
| `tools/qappswiki/cli.py` (modify) | `cmd_serve` passes the wiki root through. |
| `tools/tests/test_search.py` (new) | Splitter, BM25 ranking, cache, exclusions. |

**Phase 2 — `/home/pza/chatQEC`**

| File | Responsibility |
|---|---|
| `src/chatqec/conversation/multi_mcp.py` (new) | `MultiMCPClient`: fan out `list_tools`, route `call_tool` by name. |
| `src/chatqec/retrieval/wiki_search.py` (new) | Call `search_pages`, wrap hits as `RetrievedChunk`, never raise. |
| `src/chatqec/retrieval/scoring.py` (modify) | Two new tier weights. |
| `src/chatqec/retrieval/hybrid.py` (modify) | `retrieve(..., extra_chunks=...)` joins the rerank pool. |
| `src/chatqec/api.py` (modify) | Call the wiki searcher, pass hits to `retrieve`. |
| `src/chatqec/cli.py` (modify) | Build `MultiMCPClient` from a comma-separated server list; build the wiki searcher. |
| `conf/config.yaml` (modify) | `wiki:` block. |
| `tests/test_multi_mcp.py`, `tests/test_wiki_search.py`, `tests/test_hybrid_extra_chunks.py` (new) | One per unit. |

---

# Phase 1 — QAppsWiki

All Phase 1 work happens in `/home/pza/QAppsWiki` on `feature/mcp-search-pages`.

### Task 1: Section splitter

**Files:**
- Create: `/home/pza/QAppsWiki/tools/qappswiki/search.py`
- Test: `/home/pza/QAppsWiki/tools/tests/test_search.py`

**Interfaces:**
- Consumes: `qappswiki.parse.split_frontmatter(text) -> tuple[str | None, str]` (returns the
  raw YAML string, not a dict).
- Produces: `Section` dataclass and `split_sections(text, *, rel_path, node_id) -> list[Section]`,
  used by Tasks 2 and 3.

- [ ] **Step 1: Write the failing test**

```python
# tools/tests/test_search.py
from qappswiki.search import Section, split_sections

PAGE = """---
type: concept
name: Steane code
domains:
- quantum-error-correction
sources:
- https://errorcorrectionzoo.org/c/steane
provenance_status: needs-verification
---

# Steane code

> Imported from the Error Correction Zoo. Text reused under CC-BY-SA.

## Description

The [[7,1,3]] CSS code built from the Hamming code.

## Protection

Corrects one arbitrary single-qubit error.
"""


def test_splits_on_h2_and_carries_frontmatter():
    secs = split_sections(PAGE, rel_path="concepts/qec/steane.md", node_id="concepts/qec/steane")
    headings = [s.heading for s in secs]
    assert headings == ["(intro)", "Description", "Protection"]
    body = {s.heading: s.text for s in secs}
    assert "CSS code built from the Hamming code" in body["Description"]
    assert "single-qubit error" in body["Protection"]
    for s in secs:
        assert s.title == "Steane code"
        assert s.provenance_status == "needs-verification"
        assert s.sources == ("https://errorcorrectionzoo.org/c/steane",)
        assert s.domains == ("quantum-error-correction",)
        assert s.node_id == "concepts/qec/steane"


def test_intro_section_dropped_when_only_a_title():
    secs = split_sections("---\nname: X\n---\n\n# X\n\n## Body\n\ntext\n", rel_path="a.md", node_id="a")
    assert [s.heading for s in secs] == ["Body"]


def test_page_without_frontmatter_still_splits():
    secs = split_sections("## Only\n\nbody text\n", rel_path="a.md", node_id="a")
    assert len(secs) == 1
    assert secs[0].title == "a"            # falls back to node_id
    assert secs[0].provenance_status == ""
    assert secs[0].sources == ()


def test_section_is_hashable_and_frozen():
    s = split_sections(PAGE, rel_path="p.md", node_id="p")[0]
    assert isinstance(s, Section)
    hash(s)  # tuples, not lists, so sections can go in sets
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/pza/QAppsWiki/tools && python -m pytest tests/test_search.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'qappswiki.search'`

- [ ] **Step 3: Write minimal implementation**

```python
# tools/qappswiki/search.py
"""Stage 9: section-level full-text search over the wiki corpus.

The graph tools answer "what exists and how is it connected". This answers
"which passage says it", which is what a RAG client needs to ground and cite an
answer. Sections rather than whole pages so a hit points at the paragraph that
matched.

Pure stdlib: BM25 is ~40 lines and hand-writing it keeps the package on
pyyaml + networkx.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import yaml

from .parse import split_frontmatter

_H2 = re.compile(r"^##\s+(.*?)\s*$", re.M)
_H1 = re.compile(r"^#\s+.*$", re.M)

INTRO_HEADING = "(intro)"


@dataclass(frozen=True)
class Section:
    """One `##` block of a page, carrying its page's frontmatter."""

    rel_path: str
    node_id: str
    title: str
    heading: str
    text: str
    provenance_status: str
    sources: tuple[str, ...]
    domains: tuple[str, ...]


def _as_tuple(value) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(v) for v in value)
    return ()


def split_sections(text: str, *, rel_path: str, node_id: str) -> list[Section]:
    """Split one page into `##` sections, each carrying the page's frontmatter."""
    raw_fm, body = split_frontmatter(text)
    fm: dict = {}
    if raw_fm:
        try:
            loaded = yaml.safe_load(raw_fm)
            if isinstance(loaded, dict):
                fm = loaded
        except yaml.YAMLError:
            fm = {}

    title = str(fm.get("name") or node_id)
    provenance = str(fm.get("provenance_status") or "")
    sources = _as_tuple(fm.get("sources"))
    domains = _as_tuple(fm.get("domains"))

    def make(heading: str, chunk: str) -> Section:
        return Section(
            rel_path=rel_path,
            node_id=node_id,
            title=title,
            heading=heading,
            text=chunk.strip(),
            provenance_status=provenance,
            sources=sources,
            domains=domains,
        )

    out: list[Section] = []
    matches = list(_H2.finditer(body))

    intro = body[: matches[0].start()] if matches else body
    intro = _H1.sub("", intro).strip()
    if intro:
        out.append(make(INTRO_HEADING, intro))

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        chunk = body[m.end() : end]
        if chunk.strip():
            out.append(make(m.group(1), chunk))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/pza/QAppsWiki/tools && python -m pytest tests/test_search.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
cd /home/pza/QAppsWiki
git add tools/qappswiki/search.py tools/tests/test_search.py
git commit -m "feat(search): split pages into sections carrying their frontmatter"
```

---

### Task 2: BM25 index

**Files:**
- Modify: `/home/pza/QAppsWiki/tools/qappswiki/search.py`
- Test: `/home/pza/QAppsWiki/tools/tests/test_search.py`

**Interfaces:**
- Consumes: `Section` from Task 1.
- Produces: `Hit` dataclass (`section: Section`, `score: float`) and
  `Bm25Index(sections)` with `.search(query: str, k: int = 8, domain: str | None = None) -> list[Hit]`.
  Task 3 builds one; Task 4 queries it.

- [ ] **Step 1: Write the failing test**

```python
# append to tools/tests/test_search.py
from qappswiki.search import Bm25Index, Hit, tokenize


def _sec(node_id, heading, text, domains=("quantum-error-correction",)):
    return Section(
        rel_path=f"{node_id}.md",
        node_id=node_id,
        title=node_id,
        heading=heading,
        text=text,
        provenance_status="needs-verification",
        sources=(),
        domains=domains,
    )


def test_tokenize_lowercases_and_drops_punctuation_and_single_chars():
    assert tokenize("Surface-code THRESHOLD, p=0.01!") == ["surface", "code", "threshold", "01"]


def test_ranks_the_section_that_is_actually_about_the_query_first():
    idx = Bm25Index([
        _sec("a", "Description", "The surface code is a topological stabilizer code."),
        _sec("b", "Description", "The repetition code protects against bit flips only."),
        _sec("c", "Protection", "Surface code threshold under depolarizing noise is about 1%."),
    ])
    hits = idx.search("surface code threshold", k=3)
    assert [h.section.node_id for h in hits][0] == "c"
    assert all(isinstance(h, Hit) for h in hits)
    assert hits[0].score > hits[-1].score


def test_k_limits_results_and_zero_score_hits_are_dropped():
    idx = Bm25Index([_sec("a", "D", "alpha beta"), _sec("b", "D", "gamma delta")])
    assert len(idx.search("alpha", k=5)) == 1        # only one section matches at all
    assert idx.search("nonexistent term", k=5) == []


def test_domain_filter_excludes_other_domains():
    idx = Bm25Index([
        _sec("a", "D", "mitigation of noise", domains=("quantum-error-mitigation",)),
        _sec("b", "D", "mitigation of noise", domains=("quantum-error-correction",)),
    ])
    hits = idx.search("mitigation noise", k=5, domain="quantum-error-correction")
    assert [h.section.node_id for h in hits] == ["b"]


def test_empty_index_and_empty_query_are_safe():
    assert Bm25Index([]).search("anything", k=5) == []
    idx = Bm25Index([_sec("a", "D", "text here")])
    assert idx.search("", k=5) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/pza/QAppsWiki/tools && python -m pytest tests/test_search.py -v`
Expected: FAIL, `ImportError: cannot import name 'Bm25Index'`

- [ ] **Step 3: Write minimal implementation**

Append to `tools/qappswiki/search.py`:

```python
import math
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")
_K1 = 1.5
_B = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, dropping single characters."""
    return [t for t in _TOKEN.findall(text.lower()) if len(t) > 1]


@dataclass(frozen=True)
class Hit:
    section: Section
    score: float


class Bm25Index:
    """Okapi BM25 over section text. Built in memory, queried per request."""

    def __init__(self, sections: list[Section]) -> None:
        self.sections = list(sections)
        # Heading and title are part of the searchable text: a query naming a code
        # should match its page even when the body phrases things differently.
        self._docs = [
            Counter(tokenize(f"{s.title} {s.heading} {s.text}")) for s in self.sections
        ]
        self._lengths = [sum(d.values()) for d in self._docs]
        n = len(self._docs)
        self._avglen = (sum(self._lengths) / n) if n else 0.0
        df: Counter[str] = Counter()
        for d in self._docs:
            df.update(d.keys())
        self._idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def search(self, query: str, k: int = 8, domain: str | None = None) -> list[Hit]:
        terms = tokenize(query)
        if not terms or not self._docs:
            return []
        scored: list[Hit] = []
        for i, doc in enumerate(self._docs):
            section = self.sections[i]
            if domain and domain not in section.domains:
                continue
            length = self._lengths[i]
            total = 0.0
            for term in terms:
                freq = doc.get(term, 0)
                if not freq:
                    continue
                denom = freq + _K1 * (1 - _B + _B * length / (self._avglen or 1.0))
                total += self._idf.get(term, 0.0) * freq * (_K1 + 1) / denom
            if total > 0:
                scored.append(Hit(section=section, score=total))
        scored.sort(key=lambda h: (-h.score, h.section.node_id, h.section.heading))
        return scored[:k]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/pza/QAppsWiki/tools && python -m pytest tests/test_search.py -v`
Expected: PASS, 9 tests

- [ ] **Step 5: Commit**

```bash
cd /home/pza/QAppsWiki
git add tools/qappswiki/search.py tools/tests/test_search.py
git commit -m "feat(search): rank sections with a hand-rolled BM25"
```

---

### Task 3: Build the index over the real corpus, with caching

**Files:**
- Modify: `/home/pza/QAppsWiki/tools/qappswiki/search.py`
- Test: `/home/pza/QAppsWiki/tools/tests/test_search.py`

**Interfaces:**
- Consumes: `collect.collect_pages(root, dirs=None) -> list[dict]` where each dict is
  `{"path": abs, "rel_path": posix, "node_id": id}`; `cache.Cache(out_dir, enabled)` with
  `Cache.file_hash(path)`, `.load(digest)`, `.save(digest, payload)`.
- Produces: `build_index(root, *, out_dir=None, use_cache=True) -> Bm25Index`, called by Task 4.

- [ ] **Step 1: Write the failing test**

```python
# append to tools/tests/test_search.py
from pathlib import Path

from qappswiki.search import build_index

_PAGE_TMPL = """---
type: concept
name: {name}
domains:
- quantum-error-correction
sources: []
provenance_status: needs-verification
---

# {name}

## Description

{body}
"""


def _wiki(tmp_path: Path) -> Path:
    (tmp_path / "concepts" / "qec").mkdir(parents=True)
    (tmp_path / "concepts" / "qec" / "steane.md").write_text(
        _PAGE_TMPL.format(name="Steane code", body="A CSS code with distance three."),
        encoding="utf-8",
    )
    (tmp_path / "raw" / "md").mkdir(parents=True)
    (tmp_path / "raw" / "md" / "paper.md").write_text(
        _PAGE_TMPL.format(name="Some Paper", body="A CSS code with distance three."),
        encoding="utf-8",
    )
    return tmp_path


def test_build_index_reads_the_corpus_and_skips_raw_md(tmp_path):
    idx = build_index(_wiki(tmp_path), use_cache=False)
    hits = idx.search("CSS code distance", k=10)
    ids = {h.section.node_id for h in hits}
    assert "concepts/qec/steane" in ids
    assert not any(i.startswith("raw/md/") for i in ids)


def test_cache_round_trip_produces_identical_results(tmp_path):
    root = _wiki(tmp_path)
    out = tmp_path / "wiki-out"
    first = build_index(root, out_dir=out, use_cache=True)
    second = build_index(root, out_dir=out, use_cache=True)   # served from cache
    assert [(h.section.node_id, round(h.score, 6)) for h in first.search("CSS code", k=5)] == [
        (h.section.node_id, round(h.score, 6)) for h in second.search("CSS code", k=5)
    ]
    assert (out / "search" / "cache").is_dir()


def test_edited_page_invalidates_its_cache_entry(tmp_path):
    root = _wiki(tmp_path)
    out = tmp_path / "wiki-out"
    build_index(root, out_dir=out, use_cache=True)
    (root / "concepts" / "qec" / "steane.md").write_text(
        _PAGE_TMPL.format(name="Steane code", body="Now mentions toric lattices instead."),
        encoding="utf-8",
    )
    idx = build_index(root, out_dir=out, use_cache=True)
    assert idx.search("toric lattices", k=5)
    assert idx.search("distance three", k=5) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/pza/QAppsWiki/tools && python -m pytest tests/test_search.py -v`
Expected: FAIL, `ImportError: cannot import name 'build_index'`

- [ ] **Step 3: Write minimal implementation**

Append to `tools/qappswiki/search.py`:

```python
from pathlib import Path

from . import collect
from .cache import Cache


def _section_to_dict(s: Section) -> dict:
    return {
        "rel_path": s.rel_path,
        "node_id": s.node_id,
        "title": s.title,
        "heading": s.heading,
        "text": s.text,
        "provenance_status": s.provenance_status,
        "sources": list(s.sources),
        "domains": list(s.domains),
    }


def _section_from_dict(d: dict) -> Section:
    return Section(
        rel_path=d["rel_path"],
        node_id=d["node_id"],
        title=d["title"],
        heading=d["heading"],
        text=d["text"],
        provenance_status=d.get("provenance_status", ""),
        sources=tuple(d.get("sources") or ()),
        domains=tuple(d.get("domains") or ()),
    )


def build_index(root, *, out_dir=None, use_cache: bool = True) -> Bm25Index:
    """Split every corpus page into sections and build a BM25 index.

    `collect_pages` already drops `raw/md`, `raw/pdf`, `tools/`, and the rest of
    `paths.IGNORED_DIRS`, so exclusions live in one place.

    The cache root is `<out_dir>/search`, NOT `<out_dir>`: `Cache` keys entries on
    the file digest alone, so sharing a root with the parse cache would make the
    two overwrite each other for the same page.
    """
    root = Path(root).resolve()
    cache = None
    if use_cache and out_dir is not None:
        cache = Cache(Path(out_dir) / "search", enabled=True)

    sections: list[Section] = []
    for ref in collect.collect_pages(root):
        path = Path(ref["path"])
        digest = Cache.file_hash(path) if cache else None
        payload = cache.load(digest) if cache else None
        if payload is None:
            text = path.read_text(encoding="utf-8", errors="replace")
            page_sections = split_sections(
                text, rel_path=ref["rel_path"], node_id=ref["node_id"]
            )
            if cache:
                cache.save(digest, [_section_to_dict(s) for s in page_sections])
        else:
            page_sections = [_section_from_dict(d) for d in payload]
        sections.extend(page_sections)
    return Bm25Index(sections)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/pza/QAppsWiki/tools && python -m pytest tests/test_search.py -v`
Expected: PASS, 12 tests

- [ ] **Step 5: Sanity-check against the real 691-page corpus**

Run:
```bash
cd /home/pza/QAppsWiki/tools && python -c "
from qappswiki.search import build_index
idx = build_index('..', use_cache=False)
print('sections:', len(idx.sections))
for h in idx.search('surface code threshold under depolarizing noise', k=5):
    print(f'{h.score:6.2f}  {h.section.node_id}  ## {h.section.heading}')
"
```
Expected: a few thousand sections, and QEC-relevant pages at the top. If the top hits are
obviously wrong, stop and report rather than continuing.

- [ ] **Step 6: Commit**

```bash
cd /home/pza/QAppsWiki
git add tools/qappswiki/search.py tools/tests/test_search.py
git commit -m "feat(search): build the section index from the corpus, cached per page"
```

---

### Task 4: Expose `search_pages` over MCP

**Files:**
- Modify: `/home/pza/QAppsWiki/tools/qappswiki/serve.py`
- Modify: `/home/pza/QAppsWiki/tools/qappswiki/cli.py` (`cmd_serve`, around line 387)
- Test: `/home/pza/QAppsWiki/tools/tests/test_search.py`

**Interfaces:**
- Consumes: `build_index` from Task 3.
- Produces: `q_search_pages(index, query, k=8, domain=None) -> list[dict]` and the MCP tool
  `search_pages`. ChatQEC's Task 7 depends on this exact payload shape:
  `{path, node_id, title, heading, text, score, provenance_status, sources, url}`.

- [ ] **Step 1: Write the failing test**

```python
# append to tools/tests/test_search.py
from qappswiki.serve import q_search_pages


def test_q_search_pages_returns_the_client_payload(tmp_path):
    idx = build_index(_wiki(tmp_path), use_cache=False)
    rows = q_search_pages(idx, "CSS code distance", k=3)
    assert rows, "expected at least one hit"
    row = rows[0]
    assert set(row) == {
        "path", "node_id", "title", "heading", "text",
        "score", "provenance_status", "sources", "url",
    }
    assert row["node_id"] == "concepts/qec/steane"
    assert row["url"] == (
        "https://github.com/QSCSoftwareEcosystem/QAppsWiki/blob/main/concepts/qec/steane.md"
    )
    assert isinstance(row["score"], float)
    assert isinstance(row["sources"], list)


def test_q_search_pages_is_json_serializable(tmp_path):
    import json
    idx = build_index(_wiki(tmp_path), use_cache=False)
    json.dumps(q_search_pages(idx, "CSS", k=2))  # must not raise


def test_q_search_pages_empty_query_returns_empty(tmp_path):
    idx = build_index(_wiki(tmp_path), use_cache=False)
    assert q_search_pages(idx, "", k=3) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/pza/QAppsWiki/tools && python -m pytest tests/test_search.py -v`
Expected: FAIL, `ImportError: cannot import name 'q_search_pages'`

- [ ] **Step 3: Write minimal implementation**

Add to `tools/qappswiki/serve.py`, next to the other `q_*` functions:

```python
_BLOB_BASE = "https://github.com/QSCSoftwareEcosystem/QAppsWiki/blob/main/"


def q_search_pages(index, query, k=8, domain=None) -> list[dict]:
    """Full-text section search — the passage-level counterpart to q_query.

    q_query matches ids, titles, and domains, which answers "does this node
    exist". A RAG client needs the prose that supports an answer, so this returns
    section text plus the provenance a caller needs to cite it.
    """
    return [
        {
            "path": h.section.rel_path,
            "node_id": h.section.node_id,
            "title": h.section.title,
            "heading": h.section.heading,
            "text": h.section.text,
            "score": round(float(h.score), 6),
            "provenance_status": h.section.provenance_status,
            "sources": list(h.section.sources),
            "url": _BLOB_BASE + h.section.rel_path,
        }
        for h in index.search(query, k=k, domain=domain)
    ]
```

Then wire it into `start_server`. Change the signature and build the index lazily so the
server still starts fast:

```python
def start_server(graph_path, content_root=None):  # pragma: no cover - requires mcp + stdio
```

Immediately after `g = load_graph(graph_path)` add:

```python
    from pathlib import Path as _Path

    from .search import build_index

    root = _Path(content_root) if content_root else _Path(graph_path).resolve().parent
    _index_holder: dict = {}

    def _index():
        # Built on first search so `serve` starts instantly for graph-only clients.
        if "idx" not in _index_holder:
            _index_holder["idx"] = build_index(root, use_cache=False)
        return _index_holder["idx"]
```

Add to the `tools` list:

```python
        Tool(name="search_pages",
             description=("Full-text search over wiki page sections. Returns passage text with "
                          "provenance — use this to ground an answer, not just to find a node."),
             inputSchema={"type": "object", "properties": {
                 "query": {"type": "string"}, "k": {"type": "integer"}, "domain": {"type": "string"}},
                 "required": ["query"]}),
```

Add to the `dispatch` dict:

```python
        "search_pages": lambda a: q_search_pages(
            _index(), a["query"], a.get("k", 8), a.get("domain")
        ),
```

Finally in `cli.py`, `cmd_serve` already computes `root`; pass it through:

```python
    serve.start_server(graph_path, content_root=root)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/pza/QAppsWiki/tools && python -m pytest tests/ -v`
Expected: PASS, all tests including the pre-existing suite

- [ ] **Step 5: Verify the tool is actually served over MCP**

Run:
```bash
cd /home/pza/QAppsWiki/tools && pip install -e '.[mcp]' 2>&1 | tail -1
cd /home/pza/QAppsWiki && python -c "
from qappswiki import serve
from qappswiki.search import build_index
idx = build_index('.', use_cache=False)
rows = serve.q_search_pages(idx, 'stabilizer code distance', k=3)
for r in rows: print(round(r['score'],2), r['node_id'], '##', r['heading'])
"
```
Expected: three real QEC pages with scores.

- [ ] **Step 6: Commit and push**

```bash
cd /home/pza/QAppsWiki
git add tools/qappswiki/serve.py tools/qappswiki/cli.py tools/tests/test_search.py
git commit -m "feat(mcp): serve search_pages for passage-level retrieval"
git push
```

---

# Phase 2 — ChatQEC

All Phase 2 work happens in `/home/pza/chatQEC` on `main`.

### Task 5: `MultiMCPClient`

**Files:**
- Create: `/home/pza/chatQEC/src/chatqec/conversation/multi_mcp.py`
- Test: `/home/pza/chatQEC/tests/test_multi_mcp.py`

**Interfaces:**
- Consumes: `MCPToolDef` and the `MCPClient` Protocol from
  `chatqec.conversation.mcp_adapter` (`list_tools()`, `call_tool(name, args)`).
- Produces: `MultiMCPClient(clients)` satisfying that Protocol, so `MCPAdapter(client=...)`
  accepts it unchanged. Task 6 constructs it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multi_mcp.py
"""One adapter, several MCP servers: the Stim sibling and the wiki at once."""

import pytest

from chatqec.conversation.mcp_adapter import MCPToolDef
from chatqec.conversation.multi_mcp import MultiMCPClient


class FakeClient:
    def __init__(self, names, result=None, raises=None):
        self._names = names
        self._result = result or {"ok": True}
        self._raises = raises
        self.calls = []

    def list_tools(self):
        if self._raises:
            raise self._raises
        return [MCPToolDef(name=n, description=f"d:{n}", input_schema={}) for n in self._names]

    def call_tool(self, name, args):
        self.calls.append((name, args))
        return self._result


def test_lists_the_union_of_all_servers_tools():
    c = MultiMCPClient([FakeClient(["stim_simulate"]), FakeClient(["search_pages"])])
    assert [t.name for t in c.list_tools()] == ["stim_simulate", "search_pages"]


def test_routes_each_call_to_the_server_that_owns_the_tool():
    a, b = FakeClient(["stim_simulate"]), FakeClient(["search_pages"], result={"rows": []})
    c = MultiMCPClient([a, b])
    assert c.call_tool("search_pages", {"query": "x"}) == {"rows": []}
    assert b.calls == [("search_pages", {"query": "x"})]
    assert a.calls == []


def test_first_server_wins_a_name_collision():
    a, b = FakeClient(["dup"], result={"from": "a"}), FakeClient(["dup"], result={"from": "b"})
    c = MultiMCPClient([a, b])
    assert [t.name for t in c.list_tools()] == ["dup"]
    assert c.call_tool("dup", {}) == {"from": "a"}


def test_a_server_that_fails_to_list_is_skipped_not_fatal():
    broken = FakeClient(["never"], raises=RuntimeError("server died"))
    good = FakeClient(["stim_simulate"])
    c = MultiMCPClient([broken, good])
    assert [t.name for t in c.list_tools()] == ["stim_simulate"]


def test_unknown_tool_raises_keyerror():
    c = MultiMCPClient([FakeClient(["a"])])
    with pytest.raises(KeyError):
        c.call_tool("nope", {})


def test_no_clients_is_harmless():
    assert MultiMCPClient([]).list_tools() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_multi_mcp.py -q --no-cov`
Expected: FAIL, `ModuleNotFoundError: No module named 'chatqec.conversation.multi_mcp'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/chatqec/conversation/multi_mcp.py
"""Fan one MCPAdapter out across several MCP servers.

ChatQEC used to build exactly one StdioMCPClient, so adding the wiki server would
have cost the Stim tools. MCPAdapter depends only on the MCPClient Protocol
(list_tools/call_tool), so a client that fans out satisfies it with no change to
the adapter, the synthesizer, or the tool loop.

A server that fails to start is skipped rather than fatal: losing the wiki should
degrade answers, not break Stim.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from chatqec.conversation.mcp_adapter import MCPClient, MCPToolDef

logger = logging.getLogger(__name__)


class MultiMCPClient:
    """Presents several MCP servers as one, routing calls by tool name."""

    def __init__(self, clients: Sequence[MCPClient]) -> None:
        self._clients = list(clients)
        self._tools: list[MCPToolDef] = []
        self._owner: dict[str, MCPClient] = {}
        for client in self._clients:
            try:
                tools = client.list_tools()
            except Exception:  # a dead server must not take the others down
                logger.warning("MCP server failed to list tools; skipping", exc_info=True)
                continue
            for tool in tools:
                if tool.name in self._owner:
                    logger.warning("Duplicate MCP tool %r; keeping the first", tool.name)
                    continue
                self._owner[tool.name] = client
                self._tools.append(tool)

    def list_tools(self) -> list[MCPToolDef]:
        return list(self._tools)

    def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        client = self._owner.get(name)
        if client is None:
            raise KeyError(name)
        return client.call_tool(name, args)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/pza/chatQEC && uv run pytest tests/test_multi_mcp.py -q --no-cov`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
cd /home/pza/chatQEC
git add src/chatqec/conversation/multi_mcp.py tests/test_multi_mcp.py
git commit -m "feat(mcp): run several MCP servers behind one adapter"
```

---

### Task 6: Wiki search client and tier weights

**Files:**
- Create: `/home/pza/chatQEC/src/chatqec/retrieval/wiki_search.py`
- Modify: `/home/pza/chatQEC/src/chatqec/retrieval/scoring.py:11-18` (`TIER_WEIGHTS`)
- Test: `/home/pza/chatQEC/tests/test_wiki_search.py`

**Interfaces:**
- Consumes: the `search_pages` payload from Task 4
  (`{path, node_id, title, heading, text, score, provenance_status, sources, url}`);
  `RetrievedChunk` from `chatqec.retrieval.hybrid`.
- Produces: `wiki_chunks(client, query, *, k=8, domain=None) -> list[RetrievedChunk]`,
  called by Task 8's `ChatQEC` wiring.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_wiki_search.py
"""Wiki sections enter retrieval as ordinary chunks, and never break the answer."""

from chatqec.retrieval.scoring import TIER_WEIGHTS
from chatqec.retrieval.wiki_search import wiki_chunks

ROW = {
    "path": "concepts/qec/steane.md",
    "node_id": "concepts/qec/steane",
    "title": "Steane code",
    "heading": "Description",
    "text": "The [[7,1,3]] CSS code built from the Hamming code.",
    "score": 4.2,
    "provenance_status": "needs-verification",
    "sources": ["https://errorcorrectionzoo.org/c/steane"],
    "url": "https://github.com/QSCSoftwareEcosystem/QAppsWiki/blob/main/concepts/qec/steane.md",
}


class FakeClient:
    def __init__(self, result=None, raises=None):
        self._result = result
        self._raises = raises
        self.calls = []

    def call_tool(self, name, args):
        self.calls.append((name, args))
        if self._raises:
            raise self._raises
        return self._result


def test_maps_a_row_onto_a_retrieved_chunk():
    [chunk] = wiki_chunks(FakeClient({"rows": [ROW]}), "steane code", k=8)
    assert chunk.source_id == "qappswiki:concepts/qec/steane"
    assert chunk.chunk_id == "qappswiki:concepts/qec/steane#Description"
    assert chunk.title.startswith("Steane code")
    assert "Hamming code" in chunk.text
    assert chunk.score == 4.2
    assert chunk.payload["url"].endswith("concepts/qec/steane.md")
    assert chunk.payload["sources"] == ["https://errorcorrectionzoo.org/c/steane"]


def test_unverified_pages_are_tiered_below_blog_and_curated_ones_above():
    [unverified] = wiki_chunks(FakeClient({"rows": [ROW]}), "q")
    assert unverified.authority_tier == "wiki_unverified"
    [curated] = wiki_chunks(FakeClient({"rows": [{**ROW, "provenance_status": "source-backed"}]}), "q")
    assert curated.authority_tier == "wiki_curated"
    assert TIER_WEIGHTS["wiki_unverified"] < TIER_WEIGHTS["blog"] < TIER_WEIGHTS["wiki_curated"]
    assert TIER_WEIGHTS["wiki_curated"] < TIER_WEIGHTS["peer_reviewed"]


def test_zoo_imported_pages_carry_their_attribution_in_the_citation_title():
    # citations.py renders only `title` and `url`, so attribution has to ride on the
    # title or the reader never sees the CC-BY-SA requirement the Zoo imposes.
    [chunk] = wiki_chunks(FakeClient({"rows": [ROW]}), "q")
    assert chunk.title == "Steane code (QAppsWiki, CC-BY-SA Error Correction Zoo)"
    assert chunk.payload["attribution"] == "CC-BY-SA Error Correction Zoo"


def test_non_zoo_pages_are_only_marked_as_wiki():
    row = {**ROW, "sources": ["../OpenQEvo/README.md"]}
    [chunk] = wiki_chunks(FakeClient({"rows": [row]}), "q")
    assert chunk.title == "Steane code (QAppsWiki)"
    assert chunk.payload["attribution"] == ""


def test_passes_query_and_k_through_to_the_tool():
    client = FakeClient({"rows": []})
    wiki_chunks(client, "surface code", k=3, domain="quantum-error-correction")
    assert client.calls == [
        ("search_pages", {"query": "surface code", "k": 3, "domain": "quantum-error-correction"})
    ]


def test_a_bare_list_payload_is_accepted_too():
    assert len(wiki_chunks(FakeClient([ROW]), "q")) == 1


def test_every_failure_degrades_to_empty():
    assert wiki_chunks(None, "q") == []
    assert wiki_chunks(FakeClient(raises=TimeoutError("slow")), "q") == []
    assert wiki_chunks(FakeClient(raises=KeyError("search_pages")), "q") == []
    assert wiki_chunks(FakeClient({"rows": "not a list"}), "q") == []
    assert wiki_chunks(FakeClient({"rows": [{"junk": 1}, ROW]}), "q")[0].title == "Steane code"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/pza/chatQEC && uv run pytest tests/test_wiki_search.py -q --no-cov`
Expected: FAIL, `ModuleNotFoundError: No module named 'chatqec.retrieval.wiki_search'`

- [ ] **Step 3: Write minimal implementation**

First, `src/chatqec/retrieval/scoring.py`, extend `TIER_WEIGHTS`:

```python
TIER_WEIGHTS: dict[str, float] = {
    "canonical": 1.0,
    "peer_reviewed": 0.9,
    "preprint": 0.7,
    "wiki_curated": 0.6,      # QAppsWiki page whose claims are source-backed
    "blog": 0.5,
    "video": 0.5,
    "wiki_unverified": 0.35,  # QAppsWiki page still marked needs-verification
    "web": 0.3,
}
```

Then the new module:

```python
# src/chatqec/retrieval/wiki_search.py
"""QAppsWiki sections as retrieval candidates, fetched over MCP.

The wiki serves its own BM25 search (`search_pages`), so ChatQEC keeps no copy of
the corpus and always sees the current working tree. Hits come back as ordinary
RetrievedChunks and are reranked against Qdrant results, so the cross-encoder
decides what actually reaches the synthesizer.

Best-effort by construction: a missing server, a timeout, or a malformed payload
returns [] and the answer proceeds from the corpus alone.
"""

from __future__ import annotations

import logging
from typing import Any

from chatqec.retrieval.hybrid import RetrievedChunk
from chatqec.trace import trace

logger = logging.getLogger(__name__)

_TOOL = "search_pages"
_CURATED_STATUSES = frozenset({"source-backed", "verified"})
# Zoo-imported pages state: "Text is reused under CC-BY-SA; attribute the Error
# Correction Zoo". citations.py renders only title and url, so the attribution has
# to travel in the title or a reader never sees it.
_ZOO_HOST = "errorcorrectionzoo.org"
_ZOO_ATTRIBUTION = "CC-BY-SA Error Correction Zoo"


def _attribution(sources: list[str]) -> str:
    return _ZOO_ATTRIBUTION if any(_ZOO_HOST in str(s) for s in sources) else ""


def _tier(provenance_status: str) -> str:
    return "wiki_curated" if provenance_status in _CURATED_STATUSES else "wiki_unverified"


def _to_chunk(row: dict[str, Any]) -> RetrievedChunk | None:
    node_id = str(row.get("node_id") or "")
    text = str(row.get("text") or "")
    if not node_id or not text:
        return None
    heading = str(row.get("heading") or "")
    source_id = f"qappswiki:{node_id}"
    try:
        score = float(row.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    sources = list(row.get("sources") or [])
    attribution = _attribution(sources)
    base_title = str(row.get("title") or node_id)
    suffix = f"QAppsWiki, {attribution}" if attribution else "QAppsWiki"
    return RetrievedChunk(
        chunk_id=f"{source_id}#{heading}",
        source_id=source_id,
        text=text,
        title=f"{base_title} ({suffix})",
        authority_tier=_tier(str(row.get("provenance_status") or "")),
        score=score,
        payload={
            "url": str(row.get("url") or ""),
            "path": str(row.get("path") or ""),
            "heading": heading,
            "sources": sources,
            "attribution": attribution,
            "provenance_status": str(row.get("provenance_status") or ""),
        },
    )


def wiki_chunks(client, query: str, *, k: int = 8, domain: str | None = None) -> list[RetrievedChunk]:
    """Search QAppsWiki over MCP and return chunks. Never raises."""
    if client is None or not query:
        return []
    args: dict[str, Any] = {"query": query, "k": k, "domain": domain}
    try:
        payload = client.call_tool(_TOOL, args)
    except Exception:  # missing tool, dead server, timeout — all non-fatal
        logger.warning("QAppsWiki search failed; answering from the corpus alone", exc_info=True)
        return []

    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []
    chunks = [c for c in (_to_chunk(r) for r in rows if isinstance(r, dict)) if c is not None]
    trace("wiki.search", query=query, k=k, returned=len(chunks))
    return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/pza/chatQEC && uv run pytest tests/test_wiki_search.py -q --no-cov`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
cd /home/pza/chatQEC
git add src/chatqec/retrieval/wiki_search.py src/chatqec/retrieval/scoring.py tests/test_wiki_search.py
git commit -m "feat(retrieval): turn QAppsWiki sections into tiered retrieval chunks"
```

---

### Task 7: Rerank wiki hits against corpus hits

**Files:**
- Modify: `/home/pza/chatQEC/src/chatqec/retrieval/hybrid.py:68-175` (`retrieve`)
- Test: `/home/pza/chatQEC/tests/test_hybrid_extra_chunks.py`

**Interfaces:**
- Consumes: `RetrievedChunk` list from Task 6.
- Produces: `HybridRetriever.retrieve(..., extra_chunks: Sequence[RetrievedChunk] = ())`,
  called by Task 8.

**Why here and not in `api.py`:** `api.py` appends `web_chunks` *after* retrieval, so they
never reach the reranker. The spec promises wiki and corpus passages compete on one scale, and
the reranker lives inside `retrieve()`, so the injection has to happen there.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hybrid_extra_chunks.py
"""Externally supplied chunks compete in the same rerank pass as Qdrant hits."""

from chatqec.retrieval.canonical_injector import CanonicalIndex
from chatqec.retrieval.hybrid import HybridRetriever, RetrievedChunk


class FakeStore:
    def __init__(self, hits):
        self._hits = hits

    def dense_search(self, *, query_vector, top_k):
        return self._hits

    def sparse_search(self, *, sparse_query, top_k):
        return []


class FakeHit:
    def __init__(self, chunk_id, text, score=1.0):
        self.chunk_id = chunk_id
        self.score = score
        self.payload = {
            "text": text, "source_id": chunk_id, "title": "corpus",
            "authority_tier": "preprint",
        }


class FakeEmbedder:
    def embed_batch(self, texts):
        return [[0.1, 0.2] for _ in texts]


class FakeSparse:
    def encode_batch(self, texts):
        return []


class PickByKeyword:
    """Stands in for the cross-encoder: ranks by whether 'wiki' appears."""

    def rerank_scored(self, query, texts, top_k):
        scored = sorted(
            ((i, 10.0 if "wiki" in t else 1.0) for i, t in enumerate(texts)),
            key=lambda p: -p[1],
        )
        return scored[:top_k]


def _retriever(hits):
    return HybridRetriever(
        store=FakeStore(hits),
        embedder=FakeEmbedder(),
        sparse_encoder=FakeSparse(),
        reranker=PickByKeyword(),
        canonical_index=CanonicalIndex(entries=[]),
    )


def _wiki_chunk(text="wiki passage about codes"):
    return RetrievedChunk(
        chunk_id="qappswiki:concepts/qec/steane#Description",
        source_id="qappswiki:concepts/qec/steane",
        text=text,
        title="Steane code",
        authority_tier="wiki_unverified",
        score=4.2,
        payload={"url": "https://example.invalid/steane.md"},
    )


def test_extra_chunks_are_reranked_alongside_corpus_chunks():
    r = _retriever([FakeHit("c1", "corpus passage")])
    result = r.retrieve(queries=["codes"], top_k=2, candidate_k=10, extra_chunks=[_wiki_chunk()])
    assert [c.source_id for c in result.chunks][0].startswith("qappswiki:")
    assert len(result.chunks) == 2


def test_extra_chunks_survive_when_the_corpus_returns_nothing():
    r = _retriever([])
    result = r.retrieve(queries=["codes"], top_k=5, candidate_k=10, extra_chunks=[_wiki_chunk()])
    assert [c.source_id for c in result.chunks] == ["qappswiki:concepts/qec/steane"]


def test_extra_chunks_default_to_empty_and_change_nothing():
    r = _retriever([FakeHit("c1", "corpus passage")])
    result = r.retrieve(queries=["codes"], top_k=5, candidate_k=10)
    assert [c.chunk_id for c in result.chunks] == ["c1"]
    assert result.debug["extra_chunks"] == 0


def test_debug_records_how_many_extras_joined():
    r = _retriever([FakeHit("c1", "corpus passage")])
    result = r.retrieve(queries=["codes"], top_k=5, candidate_k=10, extra_chunks=[_wiki_chunk()])
    assert result.debug["extra_chunks"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/pza/chatQEC && uv run pytest tests/test_hybrid_extra_chunks.py -q --no-cov`
Expected: FAIL, `TypeError: retrieve() got an unexpected keyword argument 'extra_chunks'`

- [ ] **Step 3: Write minimal implementation**

In `src/chatqec/retrieval/hybrid.py`, add the parameter to `retrieve`:

```python
    def retrieve(
        self,
        *,
        queries: list[str],
        top_k: int,
        candidate_k: int,
        now: datetime | None = None,
        extra_chunks: Sequence[RetrievedChunk] = (),
    ) -> RetrievalResult:
```

Replace the early return for no hits (currently `if not hits_by_chunk:`) so external chunks
still get an answer through:

```python
        if not hits_by_chunk:
            matched_topics = match_topics(" ".join(queries), self.canonical_index)
            trace("retriever.no_hits", matched_topics=matched_topics)
            if not extra_chunks:
                return RetrievalResult(chunks=[], matched_topics=matched_topics)
            ranked = self._rerank_pool(queries[0], list(extra_chunks), top_k=top_k)
            return RetrievalResult(
                chunks=ranked,
                matched_topics=matched_topics,
                debug={"extra_chunks": len(extra_chunks), "corpus_candidates": 0},
            )
```

Replace step 5 (from `texts = [...]` through the `retrieved = [...]` comprehension) with a
pool built from both sources, reranked once:

```python
        # 5) rerank → top_k, ONCE over corpus candidates plus any external chunks
        #    (wiki sections). Scoring above is corpus-only, so external chunks are
        #    appended to the pool rather than scored on the §8.5 scale — the
        #    cross-encoder is the common yardstick that makes the comparison fair.
        pool: list[RetrievedChunk] = [
            RetrievedChunk(
                chunk_id=s.chunk_id,
                source_id=str(hits_by_chunk[s.chunk_id].get("source_id", "")),
                text=str(hits_by_chunk[s.chunk_id].get("text", "")),
                title=str(hits_by_chunk[s.chunk_id].get("title", "")),
                authority_tier=str(hits_by_chunk[s.chunk_id].get("authority_tier", "")),
                score=s.final_score,
                payload=hits_by_chunk[s.chunk_id],
            )
            for s in top_cands
        ]
        pool.extend(extra_chunks)
        retrieved = self._rerank_pool(queries[0], pool, top_k=top_k)
        top_rerank = self._last_top_rerank
```

Add the helper method to the class:

```python
    def _rerank_pool(
        self, query: str, pool: list[RetrievedChunk], *, top_k: int
    ) -> list[RetrievedChunk]:
        """Rerank a mixed pool and remember the top absolute score for the caller."""
        if not pool:
            self._last_top_rerank = 0.0
            return []
        reranked = self.reranker.rerank_scored(query, [c.text for c in pool], top_k=top_k)
        self._last_top_rerank = reranked[0][1] if reranked else 0.0
        trace("retriever.reranked", rerank_query=query, pool=len(pool))
        return [pool[i] for i, _ in reranked]
```

Initialize it in `__init__` so the attribute always exists:

```python
        self._last_top_rerank: float = 0.0
```

Finally add `"extra_chunks": len(extra_chunks)` to the returned `debug` dict, and keep
`"top_rerank_score": top_rerank` exactly as it is so the web fallback's weakness gate keeps
working.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/pza/chatQEC && uv run pytest tests/test_hybrid_extra_chunks.py tests/ -q --no-cov`
Expected: PASS, the new 4 plus all 423 existing tests

- [ ] **Step 5: Commit**

```bash
cd /home/pza/chatQEC
git add src/chatqec/retrieval/hybrid.py tests/test_hybrid_extra_chunks.py
git commit -m "feat(retrieval): rerank external chunks in the same pass as corpus hits"
```

---

### Task 8: Wire the wiki into `ask()` and the CLI

**Files:**
- Modify: `/home/pza/chatQEC/src/chatqec/api.py:97-118` (constructor), `:189-191` and `:275`
  (both `retrieve` call sites)
- Modify: `/home/pza/chatQEC/src/chatqec/cli.py:250-270` (MCP wiring)
- Modify: `/home/pza/chatQEC/conf/config.yaml`
- Test: `/home/pza/chatQEC/tests/test_api_wiki.py`

**Interfaces:**
- Consumes: `wiki_chunks` (Task 6), `retrieve(extra_chunks=...)` (Task 7),
  `MultiMCPClient` (Task 5).
- Produces: `ChatQEC(..., wiki_searcher=None, wiki_k=8)`; `CHATQEC_MCP_SERVER` accepting a
  comma-separated server list.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_wiki.py
"""ask() offers wiki sections to the reranker alongside corpus chunks."""

from unittest.mock import MagicMock

from chatqec.api import ChatQEC
from chatqec.conversation.classifier import TopicVerdict
from chatqec.conversation.rewriter import RewriteResult
from chatqec.conversation.synthesizer import SynthesisResult
from chatqec.retrieval.hybrid import RetrievalResult, RetrievedChunk


class RecordingRetriever:
    """Captures what ask() hands the retriever, which is the whole point here."""

    def __init__(self):
        self.extra_seen = None

    def retrieve(self, *, queries, top_k, candidate_k, extra_chunks=()):
        self.extra_seen = list(extra_chunks)
        return RetrievalResult(chunks=[], matched_topics=[], debug={"top_rerank_score": 0.9})


def _wiki_chunk():
    return RetrievedChunk(
        chunk_id="qappswiki:concepts/qec/steane#Description",
        source_id="qappswiki:concepts/qec/steane",
        text="The [[7,1,3]] CSS code.",
        title="Steane code (QAppsWiki, CC-BY-SA Error Correction Zoo)",
        authority_tier="wiki_unverified",
        score=4.2,
        payload={},
    )


def _bot(retriever, wiki_searcher):
    """Same MagicMock doubles as tests/test_api_chatqec.py's _build_chatqec helper."""
    classifier = MagicMock()
    classifier.classify.return_value = TopicVerdict(on_topic=True, refusal_message="")
    rewriter = MagicMock()
    rewriter.rewrite.return_value = RewriteResult(rewritten="q", variants=[])
    canonical_index = MagicMock()
    canonical_index.pages = []
    synthesizer = MagicMock()
    synthesizer.mcp_adapter = None
    synthesizer.synthesize.return_value = SynthesisResult(
        text="Answer.", citations=[], confidence="medium"
    )
    return ChatQEC(
        classifier=classifier,
        rewriter=rewriter,
        retriever=retriever,
        synthesizer=synthesizer,
        canonical_index=canonical_index,
        wiki_searcher=wiki_searcher,
    )


def test_wiki_searcher_output_reaches_the_retriever():
    retriever = RecordingRetriever()
    _bot(retriever, lambda q, k: [_wiki_chunk()]).ask("what is the steane code?")
    assert [c.source_id for c in retriever.extra_seen] == ["qappswiki:concepts/qec/steane"]


def test_no_wiki_searcher_means_no_extra_chunks():
    retriever = RecordingRetriever()
    _bot(retriever, None).ask("what is the steane code?")
    assert retriever.extra_seen == []


def test_a_raising_wiki_searcher_never_breaks_the_answer():
    def boom(q, k):
        raise TimeoutError("wiki is down")

    retriever = RecordingRetriever()
    _bot(retriever, boom).ask("what is the steane code?")   # must not raise
    assert retriever.extra_seen == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/pza/chatQEC && uv run pytest tests/test_api_wiki.py -q --no-cov`
Expected: FAIL, `TypeError: __init__() got an unexpected keyword argument 'wiki_searcher'`

- [ ] **Step 3: Write minimal implementation**

In `src/chatqec/api.py`, add constructor parameters next to the web-search ones:

```python
        wiki_searcher: Callable[[str, int], list[RetrievedChunk]] | None = None,
        wiki_k: int = 8,
```

and store them:

```python
        self.wiki_searcher = wiki_searcher
        self.wiki_k = wiki_k
```

Add the helper method:

```python
    def _wiki_chunks(self, question: str) -> list[RetrievedChunk]:
        """Wiki sections for the rerank pool. Best-effort: never breaks an answer."""
        if self.wiki_searcher is None:
            return []
        try:
            return self.wiki_searcher(question, self.wiki_k)
        except Exception:  # the wiki is a bonus source, never a dependency
            trace("wiki.search_failed", question=question)
            return []
```

At **both** `retrieve` call sites (in `ask` and in the streaming variant near line 275),
replace the call with:

```python
        wiki_extra = self._wiki_chunks(question)
        retrieval = self.retriever.retrieve(
            queries=queries, top_k=self.top_k, candidate_k=self.candidate_k,
            extra_chunks=wiki_extra,
        )
```

In `src/chatqec/cli.py`, replace the single-client construction:

```python
        from chatqec.conversation.mcp_adapter import MCPAdapter
        from chatqec.conversation.mcp_stdio import StdioMCPClient
        from chatqec.conversation.multi_mcp import MultiMCPClient

        # Comma-separated so one env var can name both the Stim sibling and the
        # wiki: CHATQEC_MCP_SERVER="chatqec-mcp-tools,qappswiki serve --root ~/QAppsWiki"
        commands = [c.strip() for c in mcp_server.split(",") if c.strip()]
        clients = []
        for command in commands:
            try:
                clients.append(StdioMCPClient(server_command=command))
            except Exception:
                logger.warning("MCP server %r failed to start; continuing without it", command)
        mcp_client = MultiMCPClient(clients)
        synthesizer.mcp_adapter = MCPAdapter(client=mcp_client)
        _mcp_client = mcp_client
```

Initialize `_mcp_client = None` before the `if mcp_server:` block, and pass the searcher into
the constructor:

```python
    from chatqec.retrieval.wiki_search import wiki_chunks

    _wiki_enabled = bool(cfg_dict.get("wiki", {}).get("enabled", False)) and _mcp_client is not None
    return ChatQEC(
        ...
        web_fallback=_web_enabled,
        wiki_searcher=(
            (lambda q, k: wiki_chunks(_mcp_client, q, k=k)) if _wiki_enabled else None
        ),
        wiki_k=int(cfg_dict.get("wiki", {}).get("k", 8)),
    )
```

Add to `conf/config.yaml` after the `mcp:` block:

```yaml
wiki:
  enabled: true          # QAppsWiki sections join retrieval when its MCP server is configured
  k: 8                   # sections requested per query
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/pza/chatQEC && uv run pytest tests/ -q --no-cov`
Expected: PASS, everything

- [ ] **Step 5: Verify end to end against the real wiki**

Run:
```bash
cd /home/pza/chatQEC
CHATQEC_MCP_SERVER="chatqec-mcp-tools,qappswiki serve --root /home/pza/QAppsWiki" \
  uv run chatqec ask "What is the concatenated Steane code?" 2>&1 | tail -30
```
Expected: the answer cites at least one `qappswiki:` source. If no wiki citation appears,
re-run with `CHATQEC_TRACE=1` and check the `wiki.search` trace line for `returned=0`.

- [ ] **Step 6: Commit**

```bash
cd /home/pza/chatQEC
git add src/chatqec/api.py src/chatqec/cli.py conf/config.yaml tests/test_api_wiki.py
git commit -m "feat(retrieval): let QAppsWiki sections compete for context in ask()"
```

---

### Task 9: Documentation and eval check

**Files:**
- Modify: `/home/pza/chatQEC/README.md`
- Modify: `/home/pza/chatQEC/docs/DESIGN_CHOICES.md`

- [ ] **Step 1: Run the eval set before judging the change**

Run: `cd /home/pza/chatQEC && uv run chatqec eval --help` then run the 50-pair set both with
and without `CHATQEC_MCP_SERVER` naming the wiki. Record recall@K and citation accuracy for
both. A drop in either means wiki sections are crowding out better corpus chunks; report the
numbers rather than silently accepting them.

- [ ] **Step 2: Document the feature in README.md**

Add to the architecture table a `QAppsWiki search` row pointing at
`src/chatqec/retrieval/wiki_search.py`, and a short subsection under the MCP section showing
the comma-separated `CHATQEC_MCP_SERVER` form with both servers, noting the wiki must be
cloned locally.

- [ ] **Step 3: Add the rationale to DESIGN_CHOICES.md**

Add `### Why QAppsWiki search lives in QAppsWiki` covering: GitHub never indexed the repo
(measured: 0 hits for `stabilizer` versus 188 in a control repo), so there is no search API to
call; ingesting into Qdrant would mean a mirror plus sync plus a second copy; the wiki serving
its own search keeps one copy and stays live; the cross-encoder is what makes cross-source
comparison fair.

- [ ] **Step 4: Commit**

```bash
cd /home/pza/chatQEC
git add README.md docs/DESIGN_CHOICES.md
git commit -m "docs: record how QAppsWiki search is wired in and why"
```
