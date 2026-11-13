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
| Pages | **60** (56 v1 roster + 4 v0.4) |
| Embedded chunks | **314** |
| Chunks per page | avg **~5.2** |
| v0.4 additions (2026-07-12) | `erasure-conversion`, `real-time-decoding`, `below-threshold-scaling`, `mid-circuit-measurement` |

The 4 new pages fill concept gaps the v0.4 papers surfaced (verified absent from the
prior 56 pages' bodies and aliases): erasure conversion (`2201.03540`), real-time /
online decoding (`2107.07505` + Riverlane), Λ below-threshold scaling (`2408.13687`,
`2207.06431`), and mid-circuit measurement (`2312.03982`). Each retrieves at ≥0.93
and force-injects on its slug/aliases.

**Try me — the 4 new canonical pages** (verified; `retrieve` shows force-injection +
the paired backing paper, `ask` synthesizes both):

```bash
# Force-injection + retrieval (cheap, no Claude)
.venv/bin/chatqec retrieve --query "converting errors into detectable erasures raises the threshold"  # canonical:erasure-conversion + [2201.03540]
.venv/bin/chatqec retrieve --query "decoder must keep up with the syndrome rate or the backlog grows"  # canonical:real-time-decoding
.venv/bin/chatqec retrieve --query "exponential logical error suppression Lambda factor below threshold"  # force-injects below-threshold-scaling
.venv/bin/chatqec retrieve --query "measuring ancilla qubits mid-circuit without collapsing the data"  # canonical:mid-circuit-measurement

# Alias hits (force-injection triggers on aliases, not just the slug)
.venv/bin/chatqec retrieve --query "erasure qubits and erasure-biased noise"          # alias -> erasure-conversion
.venv/bin/chatqec retrieve --query "online decoding and the decoding backlog problem"  # alias -> real-time-decoding

# Full cited answers (canonical definition + v0.4 paper as evidence)
.venv/bin/chatqec ask "What is erasure conversion and why does it raise the QEC threshold?"
.venv/bin/chatqec ask "What is the Lambda error-suppression factor and what does 'below threshold' mean?"
.venv/bin/chatqec ask "Why is real-time decoding a bottleneck, and how is the backlog problem addressed?"
```

### Vector store (`chatqec_chunks`, Qdrant)

| Metric | Value |
|---|---|
| Total points | **2935** |
| Canonical (`canonical`) | 314 |
| Book chapters (`peer_reviewed`) | 247 |
| arXiv papers (`preprint`) | 2043 |
| Blogs / vendor pages (`blog`) | 331 |
| Collection status | green |

Exact tier counts via Qdrant `points/count` (filter on `authority_tier`, current as of
2026-07-28 after the LightStim + Tsim ingestion below): canonical 314 · peer_reviewed
247 · preprint 2043 · blog 331 = **2935**.

**Registry totals (current, `sources/registry.jsonl`, 2026-07-28):** **123 sources /
2935 chunks** — 60 canonical, 30 arXiv, 6 book chapters (`pdf`), 25 blog/vendor
(`blog`, incl. the 15 Tsim mintlify docs pages below), 2 pinned GitHub-doc sources
(`github`, LightStim + Tsim). Reproduce with:
```bash
python3 -c "
import json
from collections import Counter
recs = [json.loads(l) for l in open('sources/registry.jsonl')]
print(len(recs), 'sources /', sum(r['chunk_count'] for r in recs), 'chunks')
print(Counter(r['type'] for r in recs))
"
```

### v0.4 corpus detail (added 2026-07-12)

**+665 chunks / 16 sources**, reproducible via `scripts/seed_corpus_v0_4.sh`.
Registry as of v0.4 (2026-07-12, now stale — see the current totals above):
**28 arXiv · 6 book chapters · 10 blog/vendor · 60 canonical = 104 sources**
(the 4 v0.4 canonical pages are registered too).

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

> **Benchmark prompt set:** [`eval/stim_qec_prompts.json`](../eval/stim_qec_prompts.json)
> covers all 5 of these tools — companion to
> [`eval/lightstim_qec_prompts.json`](../eval/lightstim_qec_prompts.json), same
> shape. It distinguishes prompts that supply a raw circuit (reliable) from
> ones that require the model to hand-write a circuit for a named code — the
> weak spot `docs/stim_getting_started_capability_check.md` already found —
> plus edge cases and cross-tool prompts, all with reference values verified
> directly against the real tools.

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
> Plan/spec: `docs/planning/plans/2026-07-12-code-params-tool.md`.

#### Questions that invoke each tool

Tools only fire when the app/CLI is started with the MCP server attached:

```bash
uv run chatqec ask --mcp-server chatqec-mcp-tools "…"                       # CLI
CHATQEC_MCP_SERVER=chatqec-mcp-tools uv run streamlit run chatqec_app/streamlit_app.py   # Streamlit
```

Without it, questions are answered from the corpus only (no live tool call). When tools
*are* available the **model decides** to call one based on the phrasing — so ask
imperatively (name the **code family + distance + error rate**; use verbs like
*simulate, decode, sweep, render, report parameters*).

| Tool | What triggers it | Example questions |
|---|---|---|
| `stim_simulate` | "simulate / run / sample this circuit", "how many detection events" | *"Simulate a distance-3 rotated surface-code memory circuit in Stim for 2000 shots and report the detection-event rate."* · *"Run this Stim circuit and show the measurement samples: `H 0\nCNOT 0 1\nM 0 1`."* |
| `stim_diagram` | "draw / render / show the circuit diagram" | *"Render the encoding circuit for the 3-qubit bit-flip code as a diagram."* · *"Show me a Stim timeline diagram of a repetition-code syndrome round."* |
| `pymatching_decode` | "decode / what's the logical error rate of this circuit" | *"Decode a distance-5 surface-code memory experiment at p=0.005 and give me its logical error rate."* · *"MWPM-decode this Stim circuit over 10,000 shots — what fraction fails?"* |
| `threshold_sweep` | "find / plot the threshold", "sweep p across distances" | *"Sweep the physical error rate for the rotated surface code at distances 3, 5, 7 and plot the logical error rate."* · *"Find the threshold of the repetition code and show the log-log plot."* |
| `code_params` | "how many physical qubits / [[n,k,d]] / stabilizers / code distance" | *"For a distance-5 rotated surface code, report [[n,k]], the stabilizer count, and the code distance."* · *"How many physical qubits encode one logical qubit in the distance-7 color code, and what's its distance?"* |

> Definitional questions ("what *is* the surface-code threshold?") stay corpus-only — no
> tool. The model reaches for a tool only when the question demands a **computed or
> rendered artifact** (a simulated number, a circuit image, a decoded rate, a threshold
> plot, or a code's parameters).

### LightStim knowledge base (docs + paper, ingested 2026-07-28)

Two sources were pinned to the same LightStim commit used by the MCP tool:

```bash
chatqec ingest --github QuTone/LightStim --commit-sha 47dd74976780192ae901d91958863cced1165354 \
  --github-paths docs/api --github-paths skills
# → ingested github:QuTone/LightStim@47dd7497: 79 chunks (12 curated .md files: docs/api/*, skills/*/SKILL.md)

chatqec ingest --arxiv 2604.21472 --full-text
# → ingested arxiv:2604.21472: 39 chunks (LaTeX source, TRUSTED id — verified against
#   CITATION.cff's preferred-citation before ingesting)
```

Both `active` in `sources/registry.jsonl`, 118 chunks embedded
(`chatqec embed --all-pending --provider gemini`). Retrieval sanity check —
`chatqec retrieve --query "What does LightStim's QECSystem class do?"` —
returns `github:QuTone/LightStim@47dd7497` chunks at the top with
similarity scores 0.71–0.85.

### LightStim via `qec_circuit_build`

A 6th tool, `qec_circuit_build`, was added to the `chatqec-mcp-tools` sibling to
cover code families plain Stim doesn't ship built-in: toric code, BB codes, and
PQRM (LightStim's construction of the 6.6.6 color code is included too, though
`code_params` already covers a Stim-native `color_code:memory_xyz` at a
different qubit-count convention — see the caveat below). It delegates to
[LightStim](https://github.com/QuTone/LightStim), pinned to an exact commit via
the sibling's optional `lightstim` extra, so the five existing tools keep
working unmodified whether or not it's installed.

| Registered tool | Purpose | Backend |
|---|---|---|
| **`qec_circuit_build`** | build a `stim.Circuit` for toric/color/BB/PQRM codes at a given distance + rounds | LightStim @ `47dd74976780192ae901d91958863cced1165354` (optional `lightstim` extra) |

**Test check-run:** sibling suite **44 passed** (`uv run pytest -q`, with the
`lightstim` extra installed; **36 passed / 6 skipped** without it — the 6
lightstim-gated integration tests skip cleanly), ruff + mypy strict clean in
both configurations.

**Functional smoke-run** (direct tool call, real pinned LightStim commit — not
a `chatqec ask` run, see the capability-check caveat below):

```text
qec_circuit_build toric_code       d=3 r=3: ok=True  num_qubits=36   num_detectors=56   num_observables=2
qec_circuit_build color_code_666   d=3 r=3: ok=True  num_qubits=13   num_detectors=18   num_observables=1
qec_circuit_build color_code_666   d=5 r=5: ok=True  num_qubits=37   num_detectors=90   num_observables=1
qec_circuit_build bb_code          d=6 r=6: ok=True  num_qubits=144  num_detectors=444  num_observables=12
qec_circuit_build pqrm             d=3 r=3: ok=True  num_qubits=25   num_detectors=40   num_observables=1
qec_circuit_build bb_code          d=7 r=7: ok=False error_message="bb_code only supports distance in [6, 10, 12]; got 7"
```

> **`bb_code` and `pqrm` only support a few discrete distances** — `{6, 10, 12}`
> and `{3, 7}` respectively — by construction of LightStim's real presets
> (trimmed further to fit this tool's own `[3, 15]` schema range). Other
> distances return `ok=False` with a clear message, not a crash.
>
> **`color_code_666` is not "LightStim-only."** `code_params`'s `color_code`
> family already wraps Stim's built-in `color_code:memory_xyz`, but reports
> **data qubits only** (`(3d²+1)/4`) where LightStim's `color_code_666`
> reports the **data+ancilla total** in its superdense layout — the two
> numbers legitimately disagree by design, not by bug.
>
**Live capability check (2026-07-28):** 4 of a 20-prompt benchmark set were run
via real `chatqec ask --mcp-server chatqec-mcp-tools` calls (`claude-sonnet-4-6`)
against the corpus above (LightStim's paper + curated docs now ingested, see
below). All 3 build/comparison prompts got correct numbers straight from the
tool — no hand-rolled circuits:

```text
"Build a distance-3 toric code memory circuit..."     → qec_circuit_build(toric_code, d=3, r=3) + stim_diagram → 56 detectors, 2 observables ✅
"I need a distance-6 BB code circuit..."               → qec_circuit_build(bb_code, d=6, r=6) → 144 qubits, 12 observables, 444 detectors ✅
"Compare a d=3 surface code (Stim) vs toric code..."   → code_params(surface_code_rotated, d=3) + qec_circuit_build(toric_code, d=3, r=3) → correctly attributed to 18 vs 8 stabilizers / k=2 vs k=1 ✅
```

See [`docs/lightstim_capability_check.md`](lightstim_capability_check.md) for
the full verdict table (incl. one edge-case prompt), raw logs, and the
remaining 16 prompts kept in
[`eval/lightstim_qec_prompts.json`](../eval/lightstim_qec_prompts.json) for
later benchmarking.

| Tool | What triggers it | Example questions |
|---|---|---|
| `qec_circuit_build` | "build / construct a toric / BB / PQRM code circuit", "distance-N color code circuit" | *"Build a distance-3 toric code memory circuit and tell me its detector count."* · *"I need a distance-6 BB code circuit — can you build it?"* |

### Tsim knowledge base (paper + GitHub docs + mintlify docs, ingested 2026-07-28)

Three source types were ingested for Tsim, all pinned/versioned as of the
same date:

```bash
chatqec ingest --arxiv 2604.01059 --full-text
# → ingested arxiv:2604.01059: 24 chunks (LaTeX source)

chatqec ingest --github QuEraComputing/tsim --commit-sha 21757967c971643a94722ddeede8d4ba16c61792 \
  --github-paths docs
# → ingested github:QuEraComputing/tsim@21757967: 6 chunks (docs/*.md)

for url in \
  https://tsim.mintlify.app/index.md \
  https://tsim.mintlify.app/tutorials/overview.md \
  https://tsim.mintlify.app/tutorials/encoding_demo.md \
  https://tsim.mintlify.app/tutorials/from_stim_to_tsim.md \
  https://tsim.mintlify.app/tutorials/magic_state_distillation.md \
  https://tsim.mintlify.app/install.md \
  https://tsim.mintlify.app/contributing.md \
  https://tsim.mintlify.app/api-reference/index.md \
  https://tsim.mintlify.app/api-reference/circuit.md \
  https://tsim.mintlify.app/api-reference/sampler.md \
  https://tsim.mintlify.app/api-reference/noise/channels.md \
  https://tsim.mintlify.app/api-reference/noise/dem.md \
  https://tsim.mintlify.app/api-reference/utils/encoder.md \
  https://tsim.mintlify.app/api-reference/utils/diagram.md \
  https://tsim.mintlify.app/api-reference/utils/program_text.md \
; do
  chatqec ingest --url "$url"
done
# → 15 mintlify docs pages total (tutorials/, api-reference/, incl. noise/ and utils/ subpages)
```

> **Do not repeat `--url` in a single `chatqec ingest` invocation.** `--url` is
> not `multiple=True` in `cli.py` — Click silently keeps only the *last*
> repeated value (`--url A --url B --url C` ingests only `C`). The shell `for`
> loop above is what was actually run for all 15 pages (confirmed via
> `sources/registry.jsonl`'s `ingested_at` timestamps: 10 of the 15 share one
> identical batch timestamp).

All `active` in `sources/registry.jsonl`, embedded via
`chatqec embed --all-pending --provider gemini`. Unlike the GitHub pin, these
mintlify `url:` sources have no version number of their own — the paper and
docs are re-ingested wholesale if Tsim's documentation changes materially,
same treatment as any other `url:`/`blog`-tier source (see
[`docs/DESIGN_CHOICES.md`](DESIGN_CHOICES.md#why-tsim-needs-no-commit-pin-for-its-code-dependency-only-for-kb-ingestion)
for why the *code* dependency below needs no commit pin even though this KB
ingestion still pins the GitHub docs to a commit).

### Tsim + GLCB via `tsim_simulate` / `glcb_visualize_url`

Two more tools were added to the `chatqec-mcp-tools` sibling. `tsim_simulate`
runs a Stim-superset circuit through
[Tsim](https://github.com/QuEraComputing/tsim) — a GPU-accelerated simulator
that extends Stim's instruction set with non-Clifford gates (`T`, `T_DAG`,
`R_X`/`R_Y`/`R_Z`, `U3`, `TPP`/`TPP_DAG`, `R_XX`/`R_YY`/`R_ZZ`/`R_PAULI`,
`CCZ`, `CCX`) via stabilizer rank decomposition — circuits plain `stim` itself
rejects outright. `glcb_visualize_url` builds a shareable link into QuEra's
Gemini Studio (https://bloqade.quera.com/studio/gemini/, which replaced the
retired Gemini Logical Circuit Builder) for a small circuit; it is pure string
encoding with no network call and no dependency on either `stim` or Tsim.

| Registered tool | Purpose | Backend |
|---|---|---|
| **`tsim_simulate`** | run a Tsim circuit (Stim syntax + non-Clifford gates) → measurement/detector samples | `bloqade-tsim>=0.1.5,<0.2` (optional `tsim` extra) |
| **`glcb_visualize_url`** | build a shareable GLCB visualization URL for a small circuit | none — pure URL encoding |

**Test check-run (2026-07-28):** sibling suite **43 passed / 6 skipped**
(`uv run pytest -q`, `tsim` extra installed, `lightstim` extra not installed —
the 6 skips are the pre-existing `lightstim`-gated integration tests, unaffected
by this change). Per file: `test_tsim_simulate` 3 (incl.
`test_t_gate_circuit_with_real_tsim` against the real installed package) ·
`test_glcb_visualize` 4 (incl. two byte-for-byte known-URL regression tests).

**Functional smoke-run** (direct tool call against the real installed
`bloqade-tsim==0.1.5`):

```text
tsim_simulate  H 0/CNOT 0 1/M 0 1 (no DETECTOR), 2000 shots:
   ok=True  shots=2000  samples_preview shape=(2000, 0)   # measured but nothing to report — no DETECTOR/OBSERVABLE_INCLUDE
tsim_simulate  RX 0/T 0/H 0/M 0/DETECTOR rec[-1], 200 shots:
   ok=True  shots=200   shape=(200, 1)  preview mix of 0/1   # non-Clifford T gate; plain stim.Circuit(...) raises ValueError: Gate not found: 'T'
tsim_simulate  H 0/M 0, 200000 shots:
   ok=False  error_message="shots > 100,000 not permitted"
tsim_simulate  FOO_BAR_NOT_A_GATE 0/M 0, 10 shots:
   ok=False  error_message="tsim error: Gate not found: 'FOO_BAR_NOT_A_GATE'"
glcb_visualize_url  H@0(col 0), CX 0->1(col 1), 2 qubits:
   ok=True  url=https://bloqade.quera.com/studio/gemini/#circuit={version:1.1,qubits:2,gates:[{gate:h,column:0,targets:[0]},{gate:cx,column:1,targets:[0,1]}],sampler:clifft}
glcb_visualize_url  H targeting qubit 5 on a 2-qubit register:
   ok=False  error_message="gate 'h' targets qubit >= qubits=2"
```

> **`tsim_simulate` always calls `Circuit.compile_detector_sampler()`
> internally**, so a circuit that only `M`easures with no `DETECTOR`/
> `OBSERVABLE_INCLUDE` gets an empty `(shots, 0)` preview — it does **not**
> return raw measurement bits the way `stim_simulate` does for the identical
> circuit text. Add a `DETECTOR` to see a result.
>
> **Tsim's detector convention is inverted from Stim's by default.**
> `use_reference_sample=False` (the tool default) reports the *actual*
> detector outcome, not a flip relative to a noiseless reference — e.g.
> `X 0\nM 0\nDETECTOR rec[-1]` reports `True` every shot by default but
> `False` every shot with `use_reference_sample=true` (matching Stim's
> always-XOR-against-reference convention). See
> [`eval/tsim_prompts.json`](../eval/tsim_prompts.json)'s
> `tool_reference_convention` prompts.

**Live capability check (2026-07-28):** 5 of a 20-prompt benchmark set
(`eval/tsim_prompts.json`) were run via real
`chatqec ask --mcp-server chatqec-mcp-tools` calls (`claude-sonnet-4-6`)
against the corpus above:

```text
"Simulate ... T 0 ... If I tried plain Stim, what would happen?"      → tsim_simulate(RX 0/T 0/H 0/M 0/DETECTOR, 200) → mix of 0/1, correctly says Stim has no T gate ✅
"Add a DETECTOR ... magic-state prep step ..." (2000 shots)            → tsim_simulate(R/R_X/T_DAG/DEPOLARIZE1/M/DETECTOR, 2000) → shape=(2000,1), mix of 0/1 ✅
"Build a GLCB link for the Bell-pair circuit ..."                      → glcb_visualize_url(h@0, cx@1) → exact reference URL ✅ (2 earlier attempts of the exact eval wording were rejected pre-tool-call by the off-topic classifier — see the capability-check doc)
"Simulate FOO_BAR_NOT_A_GATE 0/M 0 for 10 shots"                       → tsim_simulate(...) → ok=False, "Gate not found: 'FOO_BAR_NOT_A_GATE'" ✅ (the exact eval wording, byte-for-byte verified with json.load, and a milder QEC-framed paraphrase were also both rejected pre-tool-call — see the capability-check doc)
"Simulate the Bell pair ... then build a GLCB link for the same circuit" → tsim_simulate(...) + glcb_visualize_url(...) → all-zero detector (perfect correlation) + same H/CX circuit encoded ✅ (originally run in this file with an undisclosed "For a QEC circuit-tooling demo:" prefix, never disclosed as such; the literal eval wording — byte-for-byte verified with json.load — was separately re-tested directly and also passed classification with a correct result — see the capability-check doc)
```

See [`docs/tsim_glcb_capability_check.md`](tsim_glcb_capability_check.md) for
the full verdict table, raw logs, the off-topic-classifier finding, and the
remaining 15 prompts kept in
[`eval/tsim_prompts.json`](../eval/tsim_prompts.json) for later benchmarking.

| Tool | What triggers it | Example questions |
|---|---|---|
| `tsim_simulate` | "simulate with Tsim", "T/T_DAG gate", "non-Clifford", "magic state" | *"Using Tsim, simulate a circuit with a T gate that plain Stim can't run."* · *"Simulate this magic-state preparation circuit with Tsim for 2000 shots."* |
| `glcb_visualize_url` | "build/create a GLCB link", "visualize this circuit" | *"Build a GLCB visualization link for a Bell-pair circuit (QEC stabilizer demo)."* |

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

**Gate calibration (2026-07-13).** The fallback fires only when the best retrieved
chunk's **absolute cross-encoder rerank score** (`debug["top_rerank_score"]`, mxbai
`mxbai-rerank-base-v1`) falls below `web_fallback_threshold`. Measured against real
queries through the live retriever:

| Query class | `top_rerank_score` |
|---|---|
| Covered / in-corpus (surface code, KL, qLDPC, neutral atoms, lattice surgery…) | **0.52 – 0.99** |
| Genuinely-uncovered QEC specifics (exact pseudo-thresholds, ns latency budgets…) | **0.15 – 0.33** |
| Off-topic (pizza, stocks, dog training) | **< 0.03** |

The clusters separate cleanly, so the default threshold is **0.4** — it triggers a web
search for uncovered + off-topic questions while never firing for covered topics
(0.12 headroom below the lowest covered score). Tune via the `web_fallback_threshold`
constructor arg (raise it to search more aggressively, lower it to search only on the
clearest gaps). Note: a question whose *topic* is well-covered but whose *specific* answer
isn't (e.g. "cat-qubit CNOT fidelity on a specific chip" → 0.90) stays corpus-answered by
design — the curated content is relevant even if the exact figure isn't present.

### Vendor QEC tools via MCP (added 2026-10-27)

All measured through the chat pipeline on the local Muse-Glimmer model (`CHATQEC_PROVIDER=local`),
with the tool outputs below taken from `notebooks/getting_started.ipynb`. Install and the
per-tool extras are in the README section "Vendor QEC tools".

| Prompt | Tool | Result |
|---|---|---|
| *Compare PyMatching, MWPF, a lookup table and the exact MLE decoder on a distance-3 repetition code at p=0.02.* | `decoder_compare` | logical error rate: MLE 0.008 (capped at 500 shots), MWPF 0.012, table 0.012, PyMatching 0.014 (1000 shots, seed 1) |
| *Decode a d=5 surface code with 10 rounds at p=0.002 using the CUDA-Q sliding window decoder and compare it with PyMatching.* | `decoder_compare` | sliding window 0.0025 against PyMatching 0.0020 (2000 shots), GPU |
| *Add spacetime Pauli checks to this GHZ-3 OpenQASM circuit and report the overhead at a physical error rate of 0.01.* | `ibm_add_pauli_checks` | 3 to 4 qubits, 3 to 12 gates, depth 4 to 11, gamma 1.07, 6.2% of shots discarded |
| *Compile this rotation to Clifford+T at 6 digits of precision and give me the T count.* | `ibm_ft_compile` | `rz(0.3)`: 20, 42, 65 and 101 T gates at 2, 4, 6 and 10 digits; `rz(pi/4)` is exactly 1 |
| *Run a Deltakit memory experiment on a distance-3 rotated surface code at p=0.002.* | `deltakit_memory` | logical error rate 0.0013, falling to 0.0004 at d=5 and 0.0001 at d=7 |
| *Run a Deltakit experiment with leakage and the leakage-aware LCD decoder.* | none | the assistant says that is a Deltakit cloud service that is not configured, and offers the local run |
| *Run this Guppy program on the Selene emulator with 100 shots* (a Bell pair) | `guppy_run` | only `0,0` and `1,1`; with `CHATQEC_ALLOW_GUPPY_EXEC` unset the tool refuses and the assistant says how to enable it |
| *What are the parameters of BB5 codes?* | `code_params` | five codes, from arXiv:2503.22071 and 2606.06455, each with its table |

`code_params` also lists the tile (4 codes, arXiv:2504.09171), directional tile (13, arXiv:2606.19482)
and barbell (27, arXiv:2606.06062) codes. Every value is quoted from the paper, and
`tests/test_paper_codes.py` in the sibling repo re-checks each one against the ingested text.

Two behaviours worth knowing:

- `guppylang` 1.x rejects `result("c", measure(q))`; the compile error names the fix, and the
  assistant retries once with `measure(q).read()` and says what it changed.
- The off-topic classifier used to refuse "compile this circuit to Clifford+T" (2 of 3 runs); it now
  lists fault-tolerant compilation and the vendor tools as on-topic.

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
