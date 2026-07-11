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

## Statistics (last updated 2026-07-10)

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
| Total points | **1789** |
| Canonical points | 294 |
| Non-canonical (corpus) points | 1495 |
| Collection status | green |

### Source registry (`sources/registry.jsonl`)

| Type | Count |
|---|---|
| canonical_wiki | 56 |
| arxiv | 17 |
| pdf (book chapters) | 6 |
| blog | 5 |
| **Total** (all `active`) | **84** |

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
