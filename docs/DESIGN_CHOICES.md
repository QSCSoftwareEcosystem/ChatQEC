# Design Choices

Quick rationale for the model lineup and architectural picks in ChatQEC v0.1.
For exhaustive context see
[`docs/planning/specs/2026-05-29-chatqec-design.md`](planning/specs/2026-05-29-chatqec-design.md).

---

## Model selection

| Role | Default model | Fallback (HF) | Why this tier |
|---|---|---|---|
| Off-topic classifier | `claude-haiku-4-5` | `meta-llama/Llama-3.1-8B-Instruct` | Single-word output, on every query → must be cheap |
| Query rewriter | `claude-haiku-4-5` | `meta-llama/Llama-3.1-8B-Instruct` | Short JSON output, runs every turn |
| Synthesizer | `claude-sonnet-4-6` | `meta-llama/Llama-3.3-70B-Instruct` | Final cited answer — quality matters most |
| Eval judge | `claude-haiku-4-5` | `meta-llama/Llama-3.3-70B-Instruct` | Yes/no faithfulness check; precision over creativity |
| Dense embeddings | `gemini-embedding-001` | (none — local BGE/Nomic) | Strong retrieval, free tier, Matryoshka-truncatable |
| Cross-encoder reranker | `mixedbread-ai/mxbai-rerank-base-v1` | (local, no fallback) | Apache-2.0, strong BEIR scores; non-China origin (avoids BAAI/bge) |
| Figure describer (ingestion) | `meta-llama/Llama-3.2-11B-Vision-Instruct` (local GPU) | (skipped — text-only if no GPU/weights) | Describes paper/PDF figures into searchable chunks; runs offline, no per-image API cost |
| Image captioner (query) | provider-native vision — Claude (Anthropic) / `gemini-2.5-flash` (Gemini) | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | Captions user-attached images into the retrieval query; reuses the active LLM provider's multimodal path |

### Why Haiku for the classifier and rewriter

These run **on every single query**. Cost compounds fast.

- **Classifier** outputs one word (`on_topic` / `off_topic`). A 70B model would be overkill; Haiku gets ~99% accuracy at ~5% of Sonnet's price.
- **Rewriter** outputs a small JSON object (rewritten query + 2-3 variants). Length-bounded output, well-defined schema, no reasoning chain needed.
- **Conservative fallback on malformed output** — if Haiku ever emits something we can't parse, we refuse or pass through rather than spending Sonnet tokens on ambiguity. The classifier defaults to off-topic; the rewriter defaults to the raw query. Cost protection (spec §12.1).

### Why Sonnet for the synthesizer

The synthesizer composes the final cited answer from retrieved chunks. This is where:
- Citation precision matters (one wrong `[^N]` and the answer is unreliable).
- Mathematical claims must be tracked carefully (QEC questions are heavy on LaTeX-style notation).
- The model needs to *refuse politely* when evidence is thin, not invent.

Sonnet handles long-context multi-document synthesis substantially better than Haiku at acceptable cost (~$0.01–0.03/query). Opus would be marginal quality improvement at 5× the cost — not justified for v1.

### Why Haiku (not Sonnet) for the eval judge

The faithfulness judge outputs **one word** (`yes` / `no`) about whether the answer follows from the evidence. Same logic as the classifier: well-defined task, length-bounded output, no benefit to scaling up.

Running the full 50-case eval with Haiku judge costs ~$0.25; with Sonnet judge ~$2.50. We'd rather spend that budget on more eval cases.

### Why Gemini `text-embedding-004` → `gemini-embedding-001`

- **Free tier is generous.** ~1500 RPM, 30K TPM. Embedding 18 sources is a single-minute job.
- **Matryoshka representation.** The new `gemini-embedding-001` is natively 3072-dim but truncatable to 768 or 1536 with minimal quality loss. We use 768 to match Qdrant's natural sweet spot for storage and memory.
- **`text-embedding-004` was retired** by Google in late 2025; `gemini-embedding-001` is the GA replacement. ChatQEC migrated on hit and surfaces a clear error if you use the old name.

We considered OpenAI `text-embedding-3-large` but Gemini's free tier + Matryoshka clinched it. Vendor diversity (Anthropic for LLM, Google for embeddings) also reduces single-account-failure blast radius.

### Why mxbai-rerank-base-v1

The bi-encoder retrieval (Qdrant + BM25) is recall-oriented — fast over large corpora but coarse. A cross-encoder reranker that *re-scores the top-K candidates jointly with the query* substantially improves precision-at-k. This is a well-known RAG pattern.

- **mixedbread-ai/mxbai-rerank-base-v1** is open-weights (Apache-2.0), no API key, runs locally. It replaces `BAAI/bge-reranker-base` to avoid China-origin models; mxbai-rerank-v1 is DeBERTa-v3 based and a like-for-like upgrade (stronger BEIR scores).
- We picked `-base` (~280 MB) over `-large` (~1.5 GB) because base is much faster and the quality drop is small for short QEC contexts. Only the **v1** line is used — mxbai-rerank-**v2** is built on a Qwen base.
- The model is configurable via `models.reranker` in `conf/config.yaml`.

### Why two separate vision models

Image reading happens in two very different places, so they use different models.

- **Figure describer (offline ingestion).** When a paper or PDF is ingested, its figures are turned into short technical descriptions (code/circuit/lattice/decoder/plot + any axis labels) that become searchable chunks. This is a **local `meta-llama/Llama-3.2-11B-Vision-Instruct`** (~20 GB) running on the machine's GPU — it processes many figures in a batch job, so paying a per-image API price would be wasteful, and keeping it local avoids sending figures to a third party. If no GPU/weights are present the step is skipped automatically and ingestion stays text-only. Code: `src/chatqec/ingestion/figure_vision.py`.
- **Image captioner (online query).** When a user attaches an image in chat, it is captioned into 1–3 sentences that get folded into the retrieval query so the KB matches the figure. This reuses the **active LLM provider's own multimodal path** — Claude's native vision under `provider: anthropic`, `gemini-2.5-flash` under `provider: gemini` — with **`meta-llama/Llama-4-Scout-17B-16E-Instruct`** as the HuggingFace fallback. Code: `src/chatqec/conversation/vision.py`; the HF vision model is set at `models.hf.vision` in `conf/config.yaml`.

Both are US-origin open-weights on the fallback path, consistent with the compliance constraint below.

### HuggingFace fallback model choices

Constraint: **US-developed open-weights only** (Llama family, GPT-OSS). Reasoning:

- **Compliance-friendly** for collaborators in regulated environments (DOE labs, .gov).
- **Most stable HF serverless routing** — Llama models are consistently available across Together / Fireworks / Replicate / Groq partners. Some other open-weights models (Qwen, DeepSeek) have intermittent free-tier routing.

| Llama variant | Why we picked it |
|---|---|
| `Llama-3.1-8B-Instruct` | Smallest "real instruction" Llama. Cheap on HF free tier (~$0.0001/query). Classifier and rewriter don't need more. |
| `Llama-3.3-70B-Instruct` | Strongest US open-weights model with consistent HF serverless availability. ~70-85% of Sonnet quality on QEC synthesis benchmarks. |

GPT-OSS-20B/120B is interesting but routing partner availability was less reliable at v0.1 time. Easy swap later — edit `models.hf.synthesizer` in `conf/config.yaml`.

---

## Architectural choices

### Why hybrid retrieval (dense + BM25)

Two complementary failure modes:

| Query type | Dense vectors win | BM25 wins |
|---|---|---|
| "What's a topological code similar to the surface code?" | ✅ Captures semantic similarity | ❌ No exact keyword match |
| "Cite arXiv:2408.13687" | ❌ Numbers don't embed meaningfully | ✅ Exact paper ID match |
| "What is BP+OSD?" | ✅ Catches BP-OSD, belief propagation, etc. | ✅ Exact acronym hit |

Hybrid combines both with a tunable weighted sum (spec §8.5):

```
final_score = 0.5·dense + 0.3·bm25 + 0.15·tier + 0.05·recency
```

The `tier` and `recency` factors add the editorial priors — a canonical wiki page beats a 2018 blog when they're tied on semantic relevance.

### Why Qdrant

Three alternatives we considered:

| Store | Verdict |
|---|---|
| **Qdrant** ✅ | Native dense+sparse hybrid search in one query. Self-hostable. Free. Clean Python SDK. Single binary, no JVM, no Docker required (we verified). |
| **Chroma** | Less production-grade. No native hybrid search → would need separate BM25 layer. Smaller ecosystem. |
| **Weaviate** | Capable but heavier. Hybrid search exists. The Python SDK ergonomics felt clunkier for a small corpus. |
| **Pinecone** | Cloud-only, paid. Out for a self-hostable open-source tool. |
| **FAISS + manual BM25** | Most control, most plumbing. Reasonable if you outgrow Qdrant; not where you start. |

Qdrant's hybrid native handling — including sparse vectors alongside dense — saved us from writing scoring glue between two stores. The single-binary install also let us bypass the Docker-no-sudo blocker (see [`scripts/qdrant_local.sh`](../scripts/qdrant_local.sh)).

### Why MCP for tool use (not OpenAI function calling)

[MCP (Model Context Protocol)](https://modelcontextprotocol.io) is Anthropic's open spec for tool servers. Three reasons:

1. **Tool servers are reusable.** Anyone running Claude Desktop, Claude Code, or VS Code's Claude extension can install `chatqec-mcp-tools` and get the same Stim sandbox. Function-calling would be ChatQEC-specific glue.
2. **Standardized authn/discovery.** MCP defines `tools/list`, `tools/call`, schema validation. Function-calling spec is per-vendor.
3. **Future-proof.** The protocol is converging across providers (OpenAI adopted it in 2025). Building on it now means we don't have to rewrite when the next provider adds support.

Trade-off: the sibling `chatqec-mcp-tools` repo is a separate install. We accept that for the reusability gain.

### Why sandboxed subprocess (not Docker) for Stim execution

Spec §13 calls for Docker. We shipped subprocess + rlimits in v0.1 because:

- **No Docker required.** Works on machines without docker installed (like the user's machine where sudo isn't available).
- **Faster.** Process startup < 50 ms vs ~1 s for a fresh container.
- **Bounded.** `RLIMIT_AS` (memory), wall-clock timeout, output truncation, no `HOME`/`PATH` env. Network is implicitly absent unless the host enables it — Stim doesn't make network calls anyway.

Docker is v0.2. The interface is identical (`SandboxConfig` → `SandboxResult`), so swapping in a `DockerSandboxRunner` is a one-file change.

### Why a canonical wiki layer with force-injection

QEC definitions need authoritative sources. The risk with a pure-RAG approach:

- Random arXiv 2018 preprint defines "surface code" in a slightly idiosyncratic way.
- That chunk happens to rank #1 for "what is the surface code?"
- User gets a non-canonical answer.

The canonical layer is **hand-curated markdown pages** (currently 5 P0 pages, 52-page roster) that bypass the retrieval ranking for definitional queries. The `CanonicalIndex.match_topics()` checks if the query (or its rewrites) mentions any canonical topic by slug or alias; matched pages get prepended to the LLM's context block with a `[CANONICAL]` marker.

This is the same pattern used by Wikipedia's "see also" sidebar — editorial primary, retrieval secondary.

### Why QAppsWiki search lives in QAppsWiki

Not the canonical wiki above. [QAppsWiki](https://github.com/QSCSoftwareEcosystem/QAppsWiki) is a separate, externally-maintained repo (691 pages under `concepts/`: 649 Error Correction Zoo imports in `concepts/qec/`, 39 QEM pages in `concepts/qem/`, and 3 loose pages) reached only over MCP. It never enters the `[CANONICAL]` force-inject path; wiki sections compete in the ordinary rerank pool with everything else, carrying an authority tier (`wiki_curated`, `wiki_unverified`) that today is a label in the prompt rather than a ranking weight (see below).

**Why not the GitHub API.** GitHub has not indexed the repo for code search, so there is no search endpoint to call at query time. Measured 2026-08-24 with an authenticated token: `"stabilizer"` against `QSCSoftwareEcosystem/QAppsWiki` returns `total_count: 0`; the same query against `quantumlib/Stim`, used as a control, returns 188. A GitHub-search MCP tool wraps this same REST endpoint, so it would inherit this blind spot rather than work around it.

**Why not ingest into Qdrant.** This was the leading candidate until edit access to QAppsWiki was confirmed. It is workable, but it costs a mirror of the content, a freshness check against the wiki's HEAD, per-page `content_sha256` tracking, delete-before-reinsert on every update, registry rows, and a second copy of prose that already lives in a maintained repo. Every one of those is a staleness bug waiting to happen (the wiki and the mirror drift the moment someone edits a page and the sync job hasn't run yet).

**Chosen approach: the wiki serves its own search.** `tools/qappswiki/search.py` adds a hand-written Okapi BM25 index (pages split into `##` sections, cached by per-file content hash) to the MCP server that already sits next to the content, and ChatQEC calls it like any other MCP tool through `MultiMCPClient`. That keeps exactly one copy of the corpus, and freshness is free because the server reads the working tree instead of a snapshot. The trade-off is real: ChatQEC now depends on a stdio subprocess reading a local clone rather than a query against its own Qdrant, so a missing or broken clone means no wiki results, not stale ones (`wiki_search.py` never raises, so this degrades to corpus-only, matching the [error-handling table](planning/specs/2026-08-24-qappswiki-mcp-search-design.md#error-handling) in the design spec).

**Why the cross-encoder is what makes this fair.** A wiki section (a hand-edited `##` block) and a corpus chunk (math-aware-split paper prose) are different shapes of text, so their raw retrieval scores aren't directly comparable. Both go through the same `HybridRetriever` rerank pass (`retrieval/hybrid.py`'s `extra_chunks`), so the cross-encoder scores wiki prose against corpus prose on one scale before either survives to `top_k`. Without that shared rerank, one source would have to win by a hand-tuned rule instead of by which one actually answers the question.

**Known limitation: BM25 is lexical.** The wiki's search has no embeddings; it matches tokens, not meaning. A query for "biased noise" will not retrieve a page that only says "asymmetric Pauli channel," even though they're close in meaning. The cross-encoder can only rescore what BM25 hands it; it cannot rescue a section BM25 never retrieved. Retrieval-only checks against the real corpus show the wiki competing rather than dominating: for "concatenated Steane code" a QAppsWiki section ranks #1 and takes 3 of the top 8; for "bivariate bicycle code" it ranks #2 with 3 of 8; for "quantum Reed-Muller codes" the corpus wins all 8 slots.

**Be honest about the tier today: it is a label, not a weight.** `TIER_WEIGHTS` (`wiki_curated` 0.6, `wiki_unverified` 0.35) is consulted only by the candidate scoring that runs over Qdrant hits, and wiki sections bypass that scoring entirely: they are appended straight to the rerank pool, which is exactly what lets the cross-encoder judge them against corpus prose on one scale. Those two numbers are therefore inert for wiki content. The tier's only live effect is the label the synthesizer sees in its context block. Whether it should also move wiki ranking is deliberately deferred rather than overlooked, and changing it would need eval evidence, not a config edit.

The split is also near-vacuous in the corpus as it stands: 689 of the 691 `concepts/` pages carry `provenance_status: needs-verification` (2 are `source-backed`), so nearly everything would land at `wiki_unverified` even if the weights were live. It is still worth carrying for when the wiki verifies more pages.

### Why a math-aware chunker

Default chunkers (LangChain's `RecursiveCharacterTextSplitter` etc.) cut text by character count. They'll happily split `\begin{equation} p_L = A(p/p_{th})^{(d+1)/2}` in half.

A chunk like:

```
... below threshold the logical error rate satisfies
p_L = A(
```

is unusable. The chunker we ship protects:

- Inline math: `$...$`
- Display math: `$$...$$`
- Environments: `\begin{X}...\end{X}` (equation, align, gather, etc.)

It greedily emits a new chunk *only* at math-block boundaries. Slightly longer chunks in exchange for syntactically intact LaTeX — worth it for QEC content.

### Why an arXiv title-and-author resolver (§7.4)

LLMs hallucinate arXiv IDs ~14% of the time when asked for them ([early plan-time benchmark](planning/specs/2026-05-29-chatqec-design.md)). If we accepted any LLM-suggested arXiv ID directly, ~1 in 7 ingestion calls would silently pull the wrong paper.

The resolver:
1. Takes the *title keywords* and *author last names* (which LLMs hallucinate far less).
2. Searches arXiv's actual API.
3. Returns matched candidates with confidence scores.
4. Raises on ambiguity rather than guessing.

The `chatqec ingest --arxiv <id>` path *also* works but prints a warning. It's correct for human-curated lists (like your `scripts/seed_corpus_v0_2.sh`) where you verified the ID; wrong for LLM-suggested ones.

### Why `--github` ingestion pins to an exact commit, never a branch

`chatqec ingest --github <org>/<repo> --commit-sha <sha>` is the first fetcher
that pulls from a live, externally-maintained code repository rather than a
versioned paper or a URL snapshot. The risk that's specific to this source
type: a GitHub repo can rewrite its own history under you — `main` today is
not `main` next week, and a fast-moving pre-1.0 project (the motivating case
was [LightStim](https://github.com/QuTone/LightStim), not on PyPI, API not yet
stable) can change class names, module layout, or content underneath an
ingested chunk with no version bump to notice.

So `--commit-sha` is **mandatory**, validated as a 40-character hex commit
before any network call — never a branch name, never `main` — and, *for a
repo whose code dependency has no independent versioning of its own* (i.e.
it's installed straight from a git ref rather than from PyPI), the same sha
is reused as the *dependency* pin wherever the repo also ships code ChatQEC
calls (e.g. `chatqec-mcp-tools`'s `qec_circuit_build` tool, pinned to
LightStim `47dd7497...` in its `pyproject.toml`). One version string, two
consumers: the knowledge base never describes an API surface the running code
doesn't actually have, and vice versa. Bumping either without the other is a
deliberate, reviewed act — not something a routine re-ingest or dependency
update can silently do. (See the Tsim section below for the contrast case,
where the code dependency *does* have its own PyPI versioning and this
same-sha reuse doesn't apply.)

The "is a new version available" question is answered by the existing
watchlist mechanism (`sources/watchlist.yaml`'s `github:` list +
`refresh_github()`), not a bespoke checker — `chatqec watch refresh` already
polls `pushed_at` per watched repo; `QuTone/LightStim` is just one more entry
in a list that already existed for this purpose.

### Why Tsim needs no commit-pin for its code dependency (only for KB ingestion)

Tsim is the direct contrast case to the LightStim section above, and it's
worth being explicit about which half of the "pin everything" discipline
still applies and which half doesn't.

**The code dependency is a normal versioned PyPI package.** `chatqec-mcp-tools`'s
`tsim_simulate` tool depends on `bloqade-tsim>=0.1.5,<0.2` (`pyproject.toml`'s
`tsim` extra) — an ordinary semver range resolved by `uv`/`pip` from PyPI, the
same way `stim>=1.13` or `pymatching>=2.0` are. There is no `git+https://...`
URL, no commit sha anywhere in the dependency spec, and no risk of the
installed package's API silently drifting underneath a running server between
releases: a version bump is a deliberate `pyproject.toml` edit, reviewed like
any other dependency change, not something that happens by a repo's `main`
moving. This is the opposite of LightStim's situation (not on PyPI at all,
installed via `git+https://github.com/.../LightStim@<sha>`), and it's the
reason `tsim_simulate` needed none of the "pin the dependency to the exact
same sha as the KB ingestion" gymnastics `qec_circuit_build`/LightStim
required.

**Only the KB ingestion still pins a commit — because the *paper and docs*,
unlike the *code*, have no version number of their own.** `arxiv:2604.01059`
is a versioned preprint (fine as-is), but `github:QuEraComputing/tsim@21757967`
follows the same `--commit-sha`-mandatory rule as any other `--github`
ingestion (see above) — Tsim's `docs/` directory is a live, editable part of
the same fast-moving repo, and nothing about PyPI's versioning of the
*package* tells you whether the *documentation* chunk you embedded last week
still matches what's on disk today. The 15 mintlify `url:` doc pages ingested
alongside it get the *ordinary* `blog`-tier treatment every other `url:`
source gets (no commit pin, no special mechanism) — they're a hosted docs
site, not a git ref, and are re-ingested wholesale if their content changes
materially, the same way any other web source is refreshed.

In short: **pin the knowledge you can't otherwise verify is still current;
don't invent a pin for a dependency that already has real, reviewable
versioning.** Tsim needed the KB half of that discipline and none of the code
half; LightStim needed both, because for LightStim the *code* was itself
unversioned.

### Why authority tiers + recency in scoring (§8.5)

QEC sources are *not* equally trustworthy. A peer-reviewed Nature paper, a tutorial blog, and a forum post can all match a query semantically — but should not all be cited equally.

```python
TIER_WEIGHTS = {
    "canonical":     1.0,
    "peer_reviewed": 0.9,
    "preprint":      0.7,
    "blog":          0.5,
    "video":         0.5,
}
```

The `γ=0.15` weight on tier in the final score is modest enough that a highly-relevant blog still beats a marginally-relevant peer-reviewed paper — it's a tiebreaker, not a hammer. Recency (`δ=0.05`) is even smaller; it just nudges 2024 papers above 2019 ones when other signals are tied.

### Why the off-topic classifier (cost guard, §12.1)

Without a cheap pre-filter, every random question that hits the API spends Sonnet tokens. The classifier rejects "What's the capital of France?" for ~$0.0001 before Sonnet sees it.

Conservative on ambiguity by design: ambiguous queries → refuse, not "let Sonnet figure it out". This caps the worst-case cost per off-topic visitor at the classifier price.

### Why auto-fallback (Anthropic → HuggingFace) with logging

Real-world resilience. Anthropic credit runs out, accounts get locked, keys rotate. The cost of a stack trace mid-research is real.

The fallback is **safe** because:

1. **Explicit modes.** `auto` is the default, but `anthropic` and `huggingface` exist for cases where mixing would invalidate results (e.g., eval runs).
2. **Recoverable errors only.** Only credit / auth / quota / 403 / 429 trigger fall-through. Network errors and 5xx still raise — those are real outages you want to see.
3. **Every call logs which backend served it.** No silent quality drop. Watching stderr tells you immediately when Anthropic stopped working.
4. **Model name translation is explicit.** `claude-haiku-4-5` → `Llama-3.1-8B-Instruct` etc. is hardcoded in the FallbackLLMClient — visible, auditable, easy to change.

See [`src/chatqec/providers/fallback.py`](../src/chatqec/providers/fallback.py) for the implementation and [`tests/test_providers_fallback.py`](../tests/test_providers_fallback.py) for the failure-mode contracts (11 tests cover all branches).

---

## What we considered and didn't pick

| Option | Why not (yet) |
|---|---|
| **Local LLM as primary** (Ollama/llama.cpp) | Latency: a 70B model on CPU is ~30 s/query, vs 2 s for Sonnet. Useful as a third fallback in v0.2; not a primary. |
| **GPT-OSS as primary HF model** | HF serverless routing for GPT-OSS was less stable at v0.1 time. Llama-3.3-70B has more reliable free-tier coverage. Easy swap. |
| **Single-stage retrieval (no reranker)** | Recall@k is fine; precision@k is dramatically worse without cross-encoder reranking. The 100 ms reranker latency is worth it. |
| **In-memory vector store (FAISS)** | Fine until you outgrow it. Qdrant gives us persistence and hybrid for free. |
| **Tool calling without MCP** (Anthropic native only) | Locks tools to one provider. MCP lets the same Stim sandbox serve Claude Desktop, ChatQEC, and any future MCP-compatible client. |
| **Streaming the synthesizer through the fallback** | Streaming surfaces are provider-specific (Anthropic's `messages.stream` vs OpenAI-shape `stream=True`). Falls back gracefully to non-streaming on HF; v0.2 could add a unified streaming wrapper. |
| **Auto-fallback for the eval judge** | We *do* fall through, but headline eval numbers should be run with `provider: anthropic` explicitly — otherwise half the cases might be Sonnet, half Llama, and the result is meaningless. |
| **GUI-driven config** | Hydra + a YAML file is enough at this scale. A GUI is a v1.0 feature when there are 30+ knobs. |
| **Full Docker setup as v1** | Postponed: Docker is a hard dep that blocks users without admin rights (we hit this). v1 uses single-binary Qdrant + subprocess Stim. |

---

## Where to extend

If you want to add a new model, source type, or provider, the seams are:

| Goal | File to touch |
|---|---|
| New LLM provider | `src/chatqec/providers/<name>.py` + register in `_build_llm_client` |
| New source type (e.g. arXiv preprint server in a new field) | `src/chatqec/ingestion/fetchers/<name>.py` + wire in `chatqec ingest` |
| New pinned GitHub source to watch | add a `repo:` entry to `sources/watchlist.yaml`'s `github:` list — `refresh_github()` needs no code change |
| New scoring component | `src/chatqec/retrieval/scoring.py` — add the weight to `Coefficients` |
| New canonical page | `knowledge/canonical/<topic>.md` + add to `_roster.yaml` |
| New eval question subtype | `src/chatqec/eval/dataset.py` — add to `QuestionSubtype` enum + `eval/qa_set.jsonl` |
| New MCP tool | `chatqec-mcp-tools/src/chatqec_mcp_tools/tools/<name>.py` + register in `server.py` — e.g. `qec_circuit_build`, which isolates its one external, actively-changing dependency (LightStim) behind a single adapter file (`lightstim_adapter.py`) so upstream churn touches one seam, not the whole tool |
