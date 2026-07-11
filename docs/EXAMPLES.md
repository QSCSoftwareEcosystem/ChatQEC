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

## Statistics (last updated 2026-07-11)

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
| Total points | **1728** |
| Canonical (`canonical`) | 294 |
| Book chapters (`peer_reviewed`) | 247 |
| arXiv papers (`preprint`) | 1078 |
| Blogs (`blog`) | 109 |
| Collection status | green |

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

---

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
`--no-describe-figures` to skip):

```bash
.venv/bin/chatqec ingest --pdf Raw/book.pdf --pages 46-77 \
  --title "Introduction to Quantum Error Correction" --describe-figures
```

Then ask for a figure by description — retrieval matches on the injected
figure caption, not just the surrounding prose:

```bash
.venv/bin/chatqec ask "show me the surface-code lattice figure"
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
