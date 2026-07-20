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
| New scoring component | `src/chatqec/retrieval/scoring.py` — add the weight to `Coefficients` |
| New canonical page | `knowledge/canonical/<topic>.md` + add to `_roster.yaml` |
| New eval question subtype | `src/chatqec/eval/dataset.py` — add to `QuestionSubtype` enum + `eval/qa_set.jsonl` |
| New MCP tool | `chatqec-mcp-tools/src/chatqec_mcp_tools/tools/<name>.py` + register in `server.py` |
