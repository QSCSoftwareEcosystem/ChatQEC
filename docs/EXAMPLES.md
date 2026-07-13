# ChatQEC — Corpus Stats & Feature Examples

A quick reference for the current knowledge base and copy-paste commands to try
each feature. Regenerate the stats with the snippet at the bottom.

> **Prereqs for anything that hits the store:** Qdrant must be running and
> `.env` (with `GEMINI_API_KEY`) loaded. On this machine Qdrant is the **native
> binary**, not Docker:
>
> ```bash
> QDRANT__STORAGE__STORAGE_PATH="$(pwd)/qdrant_data/storage" \
>   /home/pza/.local/bin/qdrant &        # start it (REST on :6333)
> curl -s http://localhost:6333/collections   # confirm it's up
> ```
>
> Every example below assumes you first ran, from the repo root:
>
> ```bash
> cd /home/pza/chatQEC
> set -a; [ -f .env ] && . ./.env; set +a
> ```

---

## Statistics (last updated 2026-07-12)

> **Recency backfill (2026-07-12):** publication years were backfilled onto every
> arXiv figure chunk (275) and all 6 book PDFs (both `sources/registry.jsonl` and
> the `meta_year` payload — 247 book chunks). Only canonical (294) and blog (109)
> chunks remain year-less, by design (no publication year). See feature example 11.

### Canonical wiki

| Metric | Value |
|---|---|
| Pages | **56** (24 P0, 23 P1, 9 P2) |
| Embedded chunks | **294** |
| Chunks per page | min **4**, max **6**, avg **5.25** |
| Chunk text length (chars) | min 116, median 358, mean 371, max 846 |
| Distribution | 2 pages × 4 chunks · 38 pages × 5 · 16 pages × 6 |
| Smallest page | `leakage` (4 chunks) |
| Largest page | `threshold-theorem` (6 chunks) |

### Vector store (`chatqec_chunks`, Qdrant)

| Metric | Value |
|---|---|
| Total points | **2669** |
| Canonical (`canonical`) | 294 |
| Book chapters (`peer_reviewed`) | 247 |
| arXiv papers (`preprint`) | 1980 |
| Blogs / vendor pages (`blog`) | 148 |
| Collection status | green |

Exact tier counts via Qdrant `points/count` (filter on `authority_tier`): canonical
294 · peer_reviewed 247 · preprint 1980 · blog 148 = **2669**.

### v0.4 corpus detail (added 2026-07-12)

**+665 chunks / 16 sources**, reproducible via `scripts/seed_corpus_v0_4.sh`.
Registry now: **28 arXiv · 6 book chapters · 10 blog/vendor · 56 canonical = 100 sources.**

**11 fault-tolerance / QEC papers** — full LaTeX (Marker-PDF fallback where arXiv has
no source) + local Llama-3.2-Vision figure descriptions (626 text+figure chunks, 135
figures total):

| arXiv id | Paper | Chunks | Figures |
|---|---|---:|---:|
| `quant-ph/9512032` | Calderbank-Shor — Good QEC codes exist (CSS) | 17 | 1 |
| `quant-ph/9604034` | Knill-Laflamme — A theory of QEC codes | 35 | 1 |
| `quant-ph/0110143` | Dennis-Kitaev-Landahl-Preskill — Topological quantum memory | 119 | 22 |
| `quant-ph/0605138` | Bombín-Martín-Delgado — Topological quantum distillation (color codes) | 14 | 4 |
| `1209.2426` | Bravyi-Haah — Magic state distillation with low overhead | 25 | 3 |
| `1905.06903` | Litinski — Magic State Distillation: Not as Costly as You Think | 76 | 28 |
| `1310.2984` | Gottesman — FT computation with constant overhead | 43 | 3 |
| `2111.03654` | Panteleev-Kalachev — Asymptotically good qLDPC codes | 58 | 6 |
| `2207.06431` | Google Quantum AI — Suppressing errors by scaling a surface-code logical qubit | 108 | 42 |
| `2107.07505` | Ryan-Anderson (Quantinuum) — Real-time fault-tolerant QEC | 37 | 10 |
| `2312.03982` | Bluvstein (Harvard/QuEra) — Logical processor on reconfigurable atom arrays | 94 | 15 |

**5 QEC vendor / lab pages** (HTML → `blog` tier, 39 chunks):

| Source id | Vendor page | Chunks |
|---|---|---:|
| `url:14197f44` | Google Quantum AI — Making quantum error correction work (Willow) | 6 |
| `url:84fa74b6` | Google — Developer's guide to error correction (surface code) | 4 |
| `url:42acc910` | IBM Quantum — Scaling for quantum advantage (Starling FT roadmap, qLDPC) | 8 |
| `url:529d2782` | Quantinuum — Unlocking a more efficient path to fault tolerance | 12 |
| `url:282bbc44` | Quantinuum + CU Boulder — Making QEC easier | 9 |

**Try me — exercise the v0.4 additions** (all verified to retrieve the new sources).
`retrieve` is cheap (embeddings + Qdrant + reranker, no Claude); `ask` runs the full
cited answer:

```bash
# Foundational fault-tolerance / QEC papers now in the corpus
.venv/bin/chatqec retrieve --query "Knill-Laflamme error correction conditions"          # [quant-ph/9604034 + canonical]
.venv/bin/chatqec retrieve --query "topological quantum memory error threshold"          # [quant-ph/0110143 Dennis-Kitaev-Landahl-Preskill]
.venv/bin/chatqec retrieve --query "color code topological quantum distillation"         # [quant-ph/0605138 Bombín-Martín-Delgado]
.venv/bin/chatqec retrieve --query "why good quantum LDPC codes reduce overhead"         # [2111.03654 Panteleev-Kalachev]
.venv/bin/chatqec ask "What is the Knill-Laflamme condition for a quantum code to correct an error set?"
.venv/bin/chatqec ask "How does magic state distillation overhead scale, and how did Litinski cut it?"

# Magic-state distillation & overhead
.venv/bin/chatqec ask "How does magic state distillation overhead scale, and how did Litinski cut it?"         # [1905.06903 Litinski, 1209.2426 Bravyi-Haah]
.venv/bin/chatqec ask "What does 'fault tolerance with constant overhead' mean and how is it achieved?"        # [1310.2984 Gottesman]
.venv/bin/chatqec ask "Why do good quantum LDPC codes matter for the resource cost of fault tolerance?"        # [2111.03654 Panteleev-Kalachev]

# Hardware fault-tolerance demonstrations
.venv/bin/chatqec ask "Summarize the evidence that a surface-code logical qubit improves as distance grows."   # [2207.06431 Google]
.venv/bin/chatqec ask "How were logical qubits run on reconfigurable neutral-atom arrays?"                     # [2312.03982 Bluvstein/QuEra]
.venv/bin/chatqec ask "How does real-time fault-tolerant error correction work on a trapped-ion machine?"      # [2107.07505 Quantinuum]

# QEC vendor / industry info (blog tier)
.venv/bin/chatqec retrieve --query "IBM roadmap to a fault-tolerant quantum computer with qLDPC"      # [IBM qdc-2025 blog]
.venv/bin/chatqec retrieve --query "Google Willow surface code error correction below threshold"      # [Google Willow blog]
.venv/bin/chatqec ask "What is Quantinuum's stated path to fault tolerance and their logical-qubit encoding rate?"   # [Quantinuum blogs]
.venv/bin/chatqec ask "What do Google, IBM, and Quantinuum each say about their path to fault tolerance?"
```

> **Verified live (2026-07-12):** the Knill-Laflamme, neutral-atom (`2312.03982`),
> and cross-vendor `ask` queries were run end-to-end and returned cited answers at
> `confidence: high`, drawing on the newly-ingested sources (e.g. the neutral-atom
> answer grounded in `arxiv:2312.03982`; the cross-vendor answer cited the new
> Quantinuum vendor pages). On broad multi-entity questions, top-K retrieval can be
> dominated by whichever source has the most/closest chunks — narrower queries pull
> each vendor cleanly (Google `0.88`, IBM surfaced for its roadmap query).

> Note: `retrieve` blends semantic similarity with an authority-tier weight, so a
> curated **canonical** page can outrank a raw paper chunk on definitional queries
> (e.g. Knill-Laflamme) — that's by design. Ask `ask` for a synthesized answer that
> cites across both.

Embedding model: **`gemini-embedding-001`** (768-d dense) + BM25 sparse, one collection.

### Source registry (`sources/registry.jsonl`)

| Type | Count |
|---|---|
| canonical_wiki | 56 |
| arxiv | 17 |
| pdf (book chapters) | 6 |
| blog | 5 |
| **Total** (all `active`) | **84** |

### PDF book chapters & figure vision (re-ingested 2026-07-11)

The 6 book-chapter PDFs were re-ingested with figure vision. Each PDF flows
through a **three-model pipeline**:

| Stage | Model / tool | Role | Where it runs |
|---|---|---|---|
| Extract | **Marker** (`marker-pdf`, `marker_single`) | PDF → Markdown (equations→LaTeX, figures → `data/figures/<id>/`) | local GPU, isolated `uvx` env |
| Describe | **Llama-3.2-11B-Vision-Instruct** (Meta, US-origin) | per-figure description injected as `> **Figure (auto-described):**` | local GPU, bf16, `max_new_tokens=220` |
| Embed | **gemini-embedding-001** (768-d) | chunk text (incl. figure descriptions) → Qdrant | Gemini API |

Per-book results:

| Source id | Chapter | Chunks | Figures (files) | Descriptions injected¹ | Chunk words (min/med/max)² |
|---|---|---:|---:|---:|---|
| `pdf:fd245cf9:pp459-533` | Nielsen & Chuang — Ch.10 QEC | 89 | 31 | 44 | 185 / 339 / 9417 |
| `pdf:02f9b2bd` | Lidar & Brun — Intro to QEC | 54 | 7 | 7 | 53 / 354 / 818 |
| `pdf:08e4aa0f` | Lidar & Brun — Entanglement-Assisted codes | 36 | 3 | 3 | 79 / 343 / 467 |
| `pdf:af1f0de8` | Lidar & Brun — Operator QEC | 32 | 4 | 4 | 214 / 346 / 412 |
| `pdf:97a2013c:pp566-660` | Picturing Quantum Software — Ch.12 (ZX) | 25 | 110 | 101 | 68 / 270 / 19798 |
| `pdf:e31e248a` | Lidar & Brun — Intro to Fault Tolerance | 11 | 11 | 11 | 136 / 358 / 5522 |
| **Total** | | **247** | **166** | **170** | |

¹ Descriptions are injected per **image-reference occurrence** (Marker repeats a
figure link across pages), so this can exceed unique files (Nielsen & Chuang) or
fall below them when decorative images are filtered out (PQS: 110 files, 9 skipped
as decorative). ² "words" ≈ whitespace tokens, not model tokens.

> ⚠️ **Data-quality caveats to revisit:** a few chunks are pathologically large
> (PQS max ≈ 19.8k words, Nielsen ≈ 9.4k) — Marker emitted big tables/appendices
> the math-aware chunker didn't split, which risks embedding truncation. Worth a
> follow-up on chunk-size capping. Figure-description quality is good and QEC-aware
> (correctly names Steane circuits, Shor's 9-qubit code, stabilizer generators).

### arXiv papers & hybrid figure vision (re-ingested 2026-07-11)

All 17 arXiv papers were re-ingested with the **hybrid** path: the high-fidelity
**LaTeX** text is kept (equations/code preserved) and the paper's **PDF** figures
are additionally harvested, described, and appended as dedicated figure chunks —
so `arxiv:<id>` provenance is preserved and equations aren't degraded.

| Stage | Model / tool | Role |
|---|---|---|
| Text | **LaTeX source** (arXiv tarball) → math-aware chunker | equation/code-faithful body chunks |
| Figures | **Marker** (PDF) → **Llama-3.2-11B-Vision-Instruct** | persist to `data/figures/<id>/` + describe each figure |
| Embed | **gemini-embedding-001** (768-d) | all chunks → Qdrant |

Totals: **17 papers · 1354 chunks · 275 figure chunks · 279 figures persisted.**
Chunk words min/med/max: all 59 / 311 / 2509; figure chunks 59 / 154 / 202.

| Source id | Paper (short) | Chunks | Figure chunks |
|---|---|---:|---:|
| `arxiv:1208.0928` | Surface codes: practical large-scale QC | 186 | 34 |
| `arxiv:quant-ph/9705052` | Gottesman — Stabilizer Codes | 141 | 8 |
| `arxiv:1302.3428` | QEC for Quantum Memories | 140 | 20 |
| `arxiv:1808.02892` | A Game of Surface Codes | 117 | 47 |
| `arxiv:quant-ph/9906129` | FTQC with constant error rate | 106 | 4 |
| `arxiv:0905.2794` | QEC for Beginners | 100 | 24 |
| `arxiv:2101.09310` | Fusion-based quantum computation | 75 | 13 |
| `arxiv:2308.07915` | High-threshold low-overhead FT | 65 | 11 |
| `arxiv:2409.17595` | Magic state cultivation | 60 | 33 |
| `arxiv:quant-ph/0008040` | GKP — Encoding a qubit in an oscillator | 56 | 8 |
| `arxiv:quant-ph/9707021` | Kitaev — FTQC by anyons | 51 | 12 |
| `arxiv:1907.11157` | QEC: An Introductory Guide | 51 | 13 |
| `arxiv:quant-ph/9601029` | Steane — Multiple-particle interference | 47 | 6 |
| `arxiv:2107.02194` | Dynamically Generated Logical Qubits | 47 | 9 |
| `arxiv:2201.03540` | Erasure conversion for FT | 44 | 12 |
| `arxiv:2408.13687` | QEC below the surface-code threshold | 35 | 18 |
| `arxiv:quant-ph/0403025` | Bravyi–Kitaev — magic-state distillation | 33 | 3 |

> Hybrid figure harvest is opt-out: `chatqec ingest --arxiv <id> --full-text
> --no-describe-figures` keeps the fast LaTeX-only behavior.

---

### QEC MCP tools — `chatqec-mcp-tools` sibling (added 2026-07-12)

The MCP sibling exposes sandboxed QEC compute tools the assistant calls instead of
guessing numbers. Three new tools shipped this round (all Stim-backed).

| Registered tool | Purpose | Backend |
|---|---|---|
| `stim_simulate` | run a Stim circuit → measurement samples | stim 1.16.0 |
| `stim_diagram` | render a Stim circuit → SVG | stim 1.16.0 |
| **`pymatching_decode`** | MWPM-decode a Stim circuit → logical error rate | stim + pymatching 2.4.0 |
| **`threshold_sweep`** | sweep p × distance for a code family → table + log-log SVG plot | stim + pymatching + matplotlib 3.10.9 |
| **`code_params`** | report a code's [[n,k]] + stabilizer count and a Stim-computed distance | stim 1.16.0 |

**Test check-run (2026-07-12):** full sibling suite **30 passed** (`uv run pytest -q`),
ruff + mypy strict clean. Per file: `test_decoding` 2 · `test_pymatching_decode` 5 ·
`test_threshold_sweep` 5 (incl. `color_code` regression) · `test_code_params` 4 · `test_server` 4.
Main-app wiring (prompt advertises the tools + Streamlit renders their output): 4 passed.

**Functional smoke-run:**

```text
pymatching_decode: ok=True  ler=0.0350  errors=70/2000  dets=8  obs=1
threshold_sweep:   ok=True  rows=6  svg_bytes=34009
   d=3 p=0.02 ler=0.0110    d=5 p=0.02 ler=0.0010   # below threshold: larger d wins
   d=3 p=0.05 ler=0.0665    d=5 p=0.05 ler=0.0450
   d=3 p=0.10 ler=0.1895    d=5 p=0.10 ler=0.2220   # above threshold: larger d loses
code_params surface_code_rotated d=5: [[n=25, k=1]]  num_stabilizers=24  stim_distance=5
code_params color_code d=3:           [[n=7,  k=1]]  num_stabilizers=6   stim_distance=2
   # color-code stim_distance is the GRAPHLIKE distance (lower bound); true d=3
```

> **`code_params` supersedes the planned `qiskit_qec_run`.** `qiskit-qec` is unshippable
> — not on PyPI (GitHub-only, abandoned, mid-refactor at HEAD, broken against modern
> Qiskit; installable only by freezing a 2023 Qiskit-0.x stack in a separate 3.10 env
> plus source patches). Rather than carry that maintenance burden, `code_params` derives
> [[n,k]] + stabilizer count from each family's standard formula and computes the code
> distance directly with **Stim** (`shortest_graphlike_error`) — no qiskit-qec dependency.
> Plan/spec: `docs/superpowers/plans/2026-07-12-code-params-tool.md`.

## Feature examples

### 1. Semantic retrieval — paraphrase with zero keyword overlap
Proves canonical pages are found by *meaning*, not just exact terms. This query
shares no words with "neutral-atom-qec" yet surfaces it:

```bash
.venv/bin/chatqec retrieve --query "correcting errors on tweezer-array atoms" 2>/dev/null \
  | grep -iE "canonical|neutral|rydberg|score|tier" | head -10
echo "--- exit ${PIPESTATUS[0]} ---"
```

More paraphrases to try (each should pull the bracketed page):

```bash
.venv/bin/chatqec retrieve --query "measuring parity without collapsing the logical qubit"   # [syndrome-extraction]
.venv/bin/chatqec retrieve --query "why over-rotations are worse than random flips"           # [coherent-errors]
.venv/bin/chatqec retrieve --query "trading bit-flips for phase-flips on a cat qubit"          # [biased-noise / cat-codes]
.venv/bin/chatqec retrieve --query "teleporting a T gate into the computation"                 # [magic-state-injection]
```

> Tip: `retrieve` uses only Gemini embeddings + Qdrant + the reranker — **no
> Claude calls**, so it's the cheapest way to sanity-check the corpus.

### 2. Force-injection — exact slug / alias / title match
Canonical pages are also force-injected verbatim when the query literally names
them (slug, `aliases`, or title). Alias hits (not just the slug):

```bash
.venv/bin/chatqec retrieve --query "Eastin-Knill barrier and transversal gates"   # alias -> transversal-gates
.venv/bin/chatqec retrieve --query "T-count and magic budget for an algorithm"     # alias -> t-gate-count
.venv/bin/chatqec retrieve --query "SI1000 circuit level noise model"              # alias -> circuit-level-noise
```

### 3. Full answer with citations (uses Claude)
```bash
.venv/bin/chatqec ask "What is the surface code threshold under depolarizing noise?"
.venv/bin/chatqec ask "How does lattice surgery implement a logical CNOT?"
```

### 4. Multimodal — attach an image to the question
```bash
.venv/bin/chatqec ask "What code is this Tanner graph?" --image /path/to/figure.png
```

### 5. Interactive chat (REPL)
```bash
.venv/bin/chatqec chat        # /quit to exit, /clear to reset context
```

### 6. Validate the canonical wiki (no store needed)
```bash
.venv/bin/chatqec canonical --dir knowledge/canonical validate
```

### 7. Re-embed canonical pages after editing them
Idempotent. `--stage-only` writes chunk files + registry without touching Qdrant:

```bash
.venv/bin/chatqec canonical embed --stage-only         # chunks + registry only
.venv/bin/chatqec canonical embed                      # + embed into Qdrant
```

### 8. Inspect the corpus / registry
```bash
.venv/bin/chatqec corpus-stats            # chunk/token statistics
.venv/bin/chatqec sources list            # registry table
```

### 9. Figures — vision-described PDF figures, searchable and shown at answer time

`chatqec ingest --pdf` now persists every extracted figure to
`data/figures/<source_id>/`, describes each real figure with a local
**Llama-3.2-11B-Vision-Instruct** model (US-origin, runs on local GPUs, zero
API cost), and injects a `> **Figure (auto-described):** …` block into the
chunk text so the figure is semantically searchable. The figure path is
carried in chunk metadata; at answer time the CLI prints a `Figures:` block
and Streamlit renders the images inline.

Requires the optional `vision` extra and a one-time Hugging Face login (the
Llama-3.2-Vision weights are gated):

```bash
uv sync --extra vision                # or: pip install -e '.[vision]'
huggingface-cli login                  # accept the model license first
```

Ingest a book chapter with figure description (default ON — use
`--no-describe-figures` to skip). **Pass `--year`** so recency scoring and the
figure chunks get the right publication year (see section 11):

```bash
.venv/bin/chatqec ingest --pdf Raw/book.pdf --pages 46-77 \
  --title "Introduction to Quantum Error Correction" --year 2013 --describe-figures
```

**Full arXiv ingest _with_ figures (hybrid):** keep the high-fidelity LaTeX
text/equations *and* harvest the paper's figures from its PDF. The year is
filled in automatically from arXiv metadata:

```bash
.venv/bin/chatqec ingest --arxiv 2408.13687 --full-text --describe-figures
```

Then ask for a figure by description — retrieval matches on the injected
figure caption, not just the surrounding prose:

```bash
.venv/bin/chatqec ask "show me the surface-code lattice figure"
```

Verify figures actually came back with the answer (the CLI prints a `Figures:`
block listing `data/figures/...` paths):

```bash
.venv/bin/chatqec ask "show me the surface-code lattice figure" 2>/dev/null \
  | grep -iA3 "figures:"
```

#### Re-ingesting the 6 existing PDF sources to gain figures

These 6 book-chapter sources were ingested **before** this feature shipped,
so they have no figures yet. The source PDFs still exist under `Raw/`; run
these to re-ingest with figure vision enabled (titles/pages taken from the
existing registry records):

```bash
.venv/bin/chatqec ingest --pdf \
  "Raw/cambridge-core_quantum-error-correction_1Jul2026/08.1_pp_163_180_Operator_quantum_error_correction.pdf" \
  --title "Lidar & Brun — Operator Quantum Error Correction" --describe-figures   # pdf:af1f0de8, pages=all

.venv/bin/chatqec ingest --pdf \
  "Raw/quantum-computation-and-quantum-information-nielsen-chuang.pdf" \
  --pages 459-533 \
  --title "Nielsen & Chuang — Ch.10 Quantum Error-Correction" --describe-figures  # pdf:fd245cf9:pp459-533

.venv/bin/chatqec ingest --pdf \
  "Raw/cambridge-core_quantum-error-correction_1Jul2026/07.2_pp_46_77_Introduction_to_quantum_error_correction.pdf" \
  --title "Lidar & Brun — Introduction to Quantum Error Correction" --describe-figures   # pdf:02f9b2bd, pages=all

.venv/bin/chatqec ingest --pdf \
  "Raw/cambridge-core_quantum-error-correction_1Jul2026/07.5_pp_126_160_Introduction_to_quantum_fault_tolerance.pdf" \
  --title "Lidar & Brun — Introduction to Quantum Fault Tolerance" --describe-figures   # pdf:e31e248a, pages=all

.venv/bin/chatqec ingest --pdf \
  "Raw/cambridge-core_quantum-error-correction_1Jul2026/08.2_pp_181_200_Entanglement-assisted_quantum_error-correcting_codes.pdf" \
  --title "Lidar & Brun — Entanglement-Assisted Quantum Error-Correcting Codes" --describe-figures   # pdf:08e4aa0f, pages=all

.venv/bin/chatqec ingest --pdf "Raw/pqs.pdf" --pages 566-660 \
  --title "Picturing Quantum Software — Ch.12 Fault-Tolerant Quantum Computation (ZX)" --describe-figures  # pdf:97a2013c:pp566-660
```

Each re-ingest command produces a **new** `pending_review` record (a fresh
`pdf:` source id derived from content hash) — it does not overwrite the
original text-only source in place. Use `chatqec sources list` to see both
side by side, and `chatqec sources mark-retracted <id>` on the old
text-only record once you've confirmed the figure-enabled re-ingest looks
good, so retrieval doesn't surface duplicate chunks for the same chapter.

### 10. Full arXiv ingest with figures — the hybrid path end to end

The same figure vision pipeline works for arXiv papers, without giving up LaTeX
text fidelity. `--full-text` keeps the LaTeX chunks (equations verbatim);
`--describe-figures` additionally fetches the PDF, persists its figures, and
appends vision-described figure chunks. Needs the `vision` extra + a GPU; on a
CPU-only host the figure harvest is skipped and text ingest still succeeds.

```bash
.venv/bin/chatqec ingest --arxiv 2408.13687 --full-text --describe-figures
# then confirm figure chunks landed for that source:
.venv/bin/chatqec ask "describe the main architecture figure from arXiv 2408.13687" 2>/dev/null \
  | grep -iA3 "figures:"
```

### 11. Recency — recent work ranked by a 5-year time decay

Retrieval blends dense + sparse similarity with an authority-tier weight **and a
recency term**: `compute_recency` maps a chunk's publication `year` through a
5-year linear decay (this year → 1.0, ≥5 years old → 0.0, **unknown year →
neutral 0.5**). Figures inherit their source's year, so a paper's figures and
text rank consistently.

Check that a source carries a year in the store (drives its recency):

```bash
python3 - <<'PY'
from qdrant_client import QdrantClient
c = QdrantClient(url="http://localhost:6333")
pts,_ = c.scroll("chatqec_chunks", limit=3, with_payload=True,
    scroll_filter={"must":[{"key":"source_id","match":{"value":"arxiv:2408.13687"}}]})
for p in pts:
    print(p.payload["source_id"], "year=", p.payload.get("meta_year"))
PY
```

See the decay curve the scorer applies (given the current year):

```bash
python3 - <<'PY'
from datetime import datetime, UTC
from chatqec.retrieval.scoring import compute_recency
now = datetime.now(tz=UTC)
for y in (now.year, now.year-2, now.year-5, None):
    print(f"year={str(y):>5} -> recency={compute_recency(year=y, now=now):.2f}")
PY
```

> **Recency is alignment, not a blanket boost.** A 2024 figure now scores the
> same as its paper's 2024 text; a 1997 figure scores the same (low) as its
> 1997 text. If you ingest a PDF **without `--year`**, its whole source sits at
> the neutral 0.5 — fine, but it won't benefit from (or be penalised by) age.

---

## Regenerate these statistics

```bash
cd /home/pza/chatQEC
# per-page chunk counts + min/max/avg
for f in data/chunks/canonical_*.jsonl; do echo "$(grep -c . "$f") $(basename "$f" .jsonl)"; done \
  | sort -n | awk '{s+=$1;n++; if(NR==1)mn=$0} END{print "pages="n" chunks="s" avg="s/n; print "min: "mn}'
# collection totals
curl -s http://localhost:6333/collections/chatqec_chunks | grep -oE '"points_count":[0-9]+'
curl -s -X POST http://localhost:6333/collections/chatqec_chunks/points/count \
  -H 'Content-Type: application/json' \
  -d '{"filter":{"must":[{"key":"authority_tier","match":{"value":"canonical"}}]},"exact":true}'
# registry breakdown
grep -oE '"type":"[a-z_]+"' sources/registry.jsonl | sort | uniq -c
```
