<p align="center">
  <img src="docs/images/chatqec_logo_final.png" alt="ChatQEC" width="480">
</p>

An open-source research assistant for **Quantum Error Correction**.

Ask QEC questions and get answers that are either cited from a curated corpus
(papers, textbook chapters, vendor and lab pages, and a hand-reviewed canonical
wiki), computed by a real tool, or an honest refusal. The model can run 28 QEC
tools: [Stim][Stim] simulation, a dozen decoders, fault-tolerant compilation,
FTPrimitiveBench and decoder-bench circuits, QuEra's Tsim and QEC Studio, and
IBM Quantum and IQM hardware tools (calibration, mapping, preview, submit). Use
it as a Python library, a CLI, or a Streamlit web app.

[Stim]: https://github.com/quantumlib/Stim
> **Status: research prototype (v0.1.0).** Installs with one script or opens in
> GitHub Codespaces (see [Install](#install)). The public release indexes the
> 60-page canonical wiki; the full corpus is loaded from a snapshot (see
> [Corpus](#on-your-machine-linux-x86_64)). Apache-2.0.

---

## System overview

<p align="center">
  <img src="docs/images/Image Vector Editor-edited.svg" alt="ChatQEC research pipeline: a QEC question is screened and rewritten, then retrieved against the curated corpus, canonical wiki, QAppsWiki, and a web-search fallback; results are scored and reranked and handed to an LLM, which either answers directly or calls Stim, PyMatching, LightStim, Tsim, GLCB, or IBM Quantum tools via MCP before returning a cited answer." width="480">
</p>

*A question is screened and rewritten, retrieved against four sources (curated
corpus, canonical wiki, QAppsWiki, and a web-search fallback), scored and
reranked, then handed to an LLM that either answers from the retrieved
evidence or calls out to a real QEC simulator via MCP when the question needs
computation — never a guess. See [How it works](#how-it-works) below for the
detailed architecture and the query-lane diagram.*

---

## Why

Existing chat assistants hallucinate on QEC: they confuse code distances,
misquote thresholds, fabricate paper titles, and can't run a stabilizer
circuit to check their own claim. ChatQEC is built specifically for QEC, with:

- **A curated wiki layer** of hand-reviewed canonical pages (surface code,
  stabilizer formalism, CSS codes, threshold theorem, magic state distillation)
  that bypass the retrieval ranking for definitional questions — if you ask
  "what is the surface code?", you always get the canonical page first.
- **Hybrid retrieval** (dense vectors + BM25 + cross-encoder reranking) over
  a typed source registry, with authority tiers and recency in the scoring.
- **Inline citations** to source IDs the reader can click and verify.
- **28 QEC tools** (Stim, LightStim, Tsim, a dozen decoders, fault-tolerant
  compilation, QuEra Gemini and QEC Studio, IBM Quantum and IQM hardware), so
  circuit-level questions get a real computed answer, not a guess.
- **Honest refusals** when the evidence isn't in the corpus, instead of
  inventing a plausible-sounding one.

---

## Install

### In the browser: GitHub Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/QSCSoftwareEcosystem/ChatQEC?quickstart=1)

Add a `GEMINI_API_KEY` [Codespaces secret](https://github.com/settings/codespaces)
first (an `ANTHROPIC_API_KEY` is optional). The Codespace installs everything,
indexes the canonical wiki, and opens the app on port 8501. Setup takes a few
minutes the first time. Without a key it still installs; add the key and run
`uv run chatqec bootstrap`, then `bash .devcontainer/start.sh`.

### On your machine (Linux x86_64)

```bash
git clone https://github.com/QSCSoftwareEcosystem/ChatQEC && cd ChatQEC
export GEMINI_API_KEY=...          # or put it in .env; ANTHROPIC_API_KEY is optional
bash .devcontainer/setup.sh        # uv, Qdrant, the app, the CPU QEC tools, the index
bash .devcontainer/start.sh        # web app on http://localhost:8501
uv run chatqec ask --mcp-server chatqec-tools "What is the surface code?"
```

**Keys.** A Gemini key is the one requirement: retrieval embeds every question
with `gemini-embedding-001`, and Gemini also answers when no Anthropic key is set.
With `ANTHROPIC_API_KEY`, Claude answers instead. To run on your own models, set
`models.provider: local` in `conf/config.yaml` (any OpenAI-compatible server).

**Corpus.** `chatqec bootstrap` indexes the 60 canonical wiki pages shipped in
this repository. The full corpus used in the paper (arXiv papers, vendor pages,
and textbook chapters) is not redistributed here; if you have a Qdrant snapshot
of it, load it with `uv run chatqec bootstrap --force --snapshot <file-or-url>`
(or set `CHATQEC_INDEX_SNAPSHOT` before the Codespace is created).

**Tools.** `setup.sh` installs the tools with their CPU extras. CUDA-Q decoders
need a GPU (`[cudaq]`), and maximum-likelihood decoding needs Gurobi (`[mle]`);
see [Vendor QEC tools](#vendor-qec-tools-decoders-pauli-checks-fault-tolerant-compilation-deltakit-guppy).

## Quick start

[Install](#install) above sets everything up. By hand, the steps are:

```bash
uv sync                                   # the app (add --extra dev for tests)
uv tool install ./packages/chatqec-tools  # the QEC tools (add extras, see below)
uv run chatqec bootstrap                  # starts Qdrant if needed, indexes the canonical wiki
uv run chatqec ask --mcp-server chatqec-tools "What is the surface code?"
uv run streamlit run chatqec_app/streamlit_app.py   # web app; set CHATQEC_MCP_SERVER=chatqec-tools first
```

Qdrant: `bootstrap` and the web app start a local `qdrant` binary if one is on
`PATH` (`.devcontainer/setup.sh` downloads it), or run `docker compose up -d`.

Corpus: `chatqec bootstrap` embeds the 60 canonical pages from this repository.
To load a full index instead, pass a Qdrant snapshot:
`uv run chatqec bootstrap --force --snapshot <file-or-url>`. You can also grow the
corpus source by source (`chatqec ingest --arxiv <id> --full-text`,
`chatqec ingest --github <org/repo> --commit-sha <sha>`, then
`chatqec embed --all-pending`); `sources/registry.jsonl` lists every source the
full corpus contains.

### Enable QEC tool execution

The tools live in this repository under [`packages/chatqec-tools`](packages/chatqec-tools)
as their own uv project, so their heavy extras (CUDA-Q, the Qiskit pins) never enter the
app's environment. Install them as a uv tool from the repository root, then point
`--mcp-server` at the `chatqec-tools` command:

```bash
uv tool install --force ./packages/chatqec-tools

uv run chatqec ask --mcp-server chatqec-tools \
  "Simulate a 3-qubit bit-flip repetition-code syndrome circuit in Stim for 2000 \
   shots and report the syndrome distribution."
```

The install gives two commands. `chatqec-tools` is what ChatQEC uses: it runs every tool
call in a fresh process (`chatqec-tools list`, `chatqec-tools call <name>` with JSON
arguments on stdin), so a timeout kills only that call and a crash in a vendor decoder
cannot break the session. `chatqec-mcp-tools` is the same tools as a stdio MCP server, for
Claude Desktop, Claude Code, or any other MCP client; ChatQEC still accepts it in
`--mcp-server`. `--mcp-server` takes a comma-separated list, so the tools and the
QAppsWiki server combine as `--mcp-server "chatqec-tools,qappswiki serve --root ~/QAppsWiki"`
(or `CHATQEC_MCP_SERVER` for the Streamlit app).

> **The installed command is a snapshot.** `uv tool install` freezes the tools at
> install time, so **after adding or changing a tool under `packages/chatqec-tools` you
> must reinstall** (with the extras you use) or the change won't be visible:
>
> ```bash
> uv tool install --force "./packages/chatqec-tools[ibm,decoders]"
> ```
>
> Check what tools the model can actually see:
>
> ```bash
> uv run python -c "from chatqec.cli import _build_chatqec; \
>   b=_build_chatqec(mcp_server='chatqec-tools'); \
>   print([t.name for t in b.synthesizer.mcp_adapter.client.list_tools()])"
> # → 28 names: stim_simulate, stim_diagram, ..., ibm_*, iqm_*, qec_studio_*
> ```

The model **decides on its own** when a question needs a computation. Each call
runs in its own process with a timeout, and the circuit tools (`stim_simulate`,
`stim_diagram`, `pymatching_decode`, `threshold_sweep`, `code_params`,
`qec_circuit_build`, `tsim_simulate`) also run in a sandbox with memory and
wall-clock limits. Reported numbers come from the tool's output, not the
model's guess.

| Group | Tools |
|---|---|
| Circuits and decoding (11) | `stim_simulate`, `stim_diagram`, `glcb_visualize_url`, `pymatching_decode`, `threshold_sweep`, `code_params`, `qec_circuit_build`, `tsim_simulate`, `qec_studio_examples`, `qec_studio_compile`, `qec_studio_link` |
| IBM Quantum (7) | `ibm_list_backends`, `ibm_backend_calibration`, `ibm_map_circuit`, `ibm_preview_execution`, `ibm_submit_execution`, `ibm_add_pauli_checks`, `ibm_ft_compile` |
| IQM (6) | `iqm_list_backends`, `iqm_backend_calibration`, `iqm_map_circuit`, `iqm_preview_execution`, `iqm_submit_execution`, `iqm_job_result` |
| External QEC packages (4) | `decoder_compare`, `ft_primitive_simulate`, `deltakit_memory`, `guppy_run` |

> **`qec_circuit_build` needs the optional `lightstim` extra.**
> The tool is always registered, but without `lightstim` installed it returns
> `ok=False` for every call. Reinstall with the extra to actually use it:
> `uv tool install --force "./packages/chatqec-tools[lightstim]"`. It builds circuits
> for code families Stim doesn't ship built-in (toric code, BB codes, PQRM,
> plus LightStim's own construction of the 6.6.6 color code) — see
> [`docs/EXAMPLES.md`](docs/EXAMPLES.md#lightstim-via-qec_circuit_build) and
> [`docs/lightstim_capability_check.md`](docs/lightstim_capability_check.md).

> **`tsim_simulate` needs the optional `tsim` extra.** Like
> `qec_circuit_build`, the tool is always registered but returns `ok=False`
> for every call without it. Reinstall with the extra to actually use it:
> `uv tool install --force "./packages/chatqec-tools[tsim]"`. Unlike `lightstim`,
> `tsim` is a normal versioned PyPI dependency (`bloqade-tsim>=0.1.5,<0.2`,
> no commit-pin needed — see
> [`docs/DESIGN_CHOICES.md`](docs/DESIGN_CHOICES.md#why-tsim-needs-no-commit-pin-for-its-code-dependency-only-for-kb-ingestion)).
> It runs Stim-superset circuits — including non-Clifford gates (`T`,
> `T_DAG`, `R_X`/`R_Y`/`R_Z`, ...) plain Stim rejects outright — via
> [Tsim](https://github.com/QuEraComputing/tsim). `glcb_visualize_url` (no
> extra needed) builds a shareable QuEra Gemini Studio link
> for a small circuit. See
> [`docs/EXAMPLES.md`](docs/EXAMPLES.md#tsim--glcb-via-tsim_simulate--glcb_visualize_url)
> and [`docs/tsim_glcb_capability_check.md`](docs/tsim_glcb_capability_check.md).

**Provider:** tool use works with the `anthropic`, `gemini`, `auto`, and `local`
providers (set `models.provider` in `conf/config.yaml`, or `CHATQEC_PROVIDER`).
Under `auto`, ChatQEC tries Anthropic first and **automatically fails over to
Gemini** on credit/auth errors. (The `huggingface` provider has no tool-use
support.)

### Vendor QEC tools (decoders, Pauli checks, fault-tolerant compilation, Deltakit, Guppy)

The same tools package also exposes the vendor tools below. Each one lives
behind an optional extra and returns `ok=False` with an install hint when its extra is
missing, so the base install keeps working. Install everything with:

```bash
uv tool install --force "./packages/chatqec-tools[lightstim,tsim,ibm,decoders,mle,cudaq,paulice,ft,deltakit,guppy]"
```

| Tool | What it does | Extra | Notes |
|---|---|---|---|
| `decoder_compare` | same syndromes through several decoders, ranked by logical error rate | `decoders`, `mle`, `cudaq` | PyMatching, Tesseract, BP+OSD/LSD, BP+union-find, **MWPF**, **MLE**, **lookup table**, CUDA-Q RelayBP, BP+OSD and **sliding window** (GPU) |
| `ibm_add_pauli_checks` | spacetime Pauli checks (IBM `qiskit-paulice`) on a Clifford circuit, with the overhead | `paulice` | qubits, gates, depth, sampling cost, and the postselection discard rate when `noise_p` is given |
| `ibm_ft_compile` | Qiskit fault-tolerant compilation: Clifford+T (gridsynth) or Pauli-based computation | `ft` | needs Qiskit 2.5 or newer; reports the T count or the rotation count |
| `deltakit_memory` | Riverlane Deltakit memory experiments decoded with PyMatching | `deltakit` | runs fully locally; see the note below |
| `guppy_run` | run a Guppy program on the Selene emulator | `guppy` | **off by default**; see the warning below |
| `code_params` | now also BB5, tile, directional tile and barbell codes | none | values quoted from the papers with their arXiv source, not computed |

> **`mle` uses Gurobi.** The pip wheel ships a size-limited licence that is for
> non-production use only, which is enough for small codes. Larger models, or production
> use, need your own Gurobi licence.

> **`deltakit_memory` is local only.** In Deltakit 0.9 and 0.10 only the PyMatching decoder runs
> locally. Its MWPM, BP+OSD, BeliefMatching, Local Clustering (the leakage-aware LCD),
> Ambiguity Clustering and Collision Clustering decoders, and leakage simulation, are cloud
> services that need an account and can cost money. The tool refuses them with "not configured".

> **`guppy_run` executes a program supplied through chat, and there is no OS sandbox.** Unprivileged
> user namespaces are blocked on many machines, so it is protected in three other ways: it is
> disabled unless ChatQEC (or the MCP server) is started with `CHATQEC_ALLOW_GUPPY_EXEC=1`; a source gate accepts
> only `guppylang` imports and `@guppy` functions (no `import os`, top-level calls, `py(...)`, or
> dunder access); and the program runs in a separate process with a scrubbed environment, a private
> temporary directory, CPU, file and memory limits, and a hard time limit. It does not stop a bug
> inside the Guppy compiler, and it does not block network access. Enable it only for programs you
> would be willing to run, for example
> `CHATQEC_ALLOW_GUPPY_EXEC=1 uv run streamlit run chatqec_app/streamlit_app.py`.

> **Python version.** `uv tool install` picks your newest Python (3.13 on a current machine), and
> guppylang 1.x needs 3.12 or newer, so you get different package versions from the 3.11 development
> environment (guppylang 1.2 against 0.21; in 1.x a measurement must be read with
> `measure(q).read()`). The tool suite is verified on both Python 3.11 and 3.13.

> **Running next to local vLLM models.** Two things break the app on a shared GPU box.
> (1) An expired `HF_TOKEN` exported in your shell (for example in `~/.bashrc`) is not overridden by
> `.env`, and `huggingface_hub` then answers 401 even for the public, cached reranker; refresh the
> token or start the app with `HF_HUB_OFFLINE=1`. (2) The reranker and the CUDA-Q decoders default
> to GPU 0, which a served model can fill (`CUDA out of memory`, and CUDA-Q prints its error on
> stdout); pin them to a free GPU with `CUDA_VISIBLE_DEVICES=<n>`.

### Quantum hardware tools (IBM Quantum, IQM)

The `ibm_*` tools need the `ibm` extra and `QISKIT_IBM_TOKEN` (optionally
`QISKIT_IBM_INSTANCE`). The `iqm_*` tools run IQM calls in a separate Python
environment, because `iqm-client` pins versions that conflict with the other
extras; create it once, then set `IQM_TOKEN` and `IQM_SERVER_URL`:

```bash
bash packages/chatqec-tools/scripts/setup_iqm_env.sh   # or set CHATQEC_IQM_PYTHON
```

Listing, calibration, mapping, and preview are free. Submission is a separate
tool that the model may call only after the user has seen the preview and asked
to run, and it uses the configured account's allocation (IQM jobs are capped at
10 qubits and 4,000 shots). [`notebooks/hardware_getting_started.ipynb`](notebooks/hardware_getting_started.ipynb)
walks through both.

### Enable QAppsWiki search

ChatQEC can pull prose sections from [QAppsWiki](https://github.com/QSCSoftwareEcosystem/QAppsWiki)
(780 concept pages, about 700 of them linked to the Error Correction Zoo) into the same
candidate pool as the corpus, by naming a **second** MCP server alongside the
QEC tools. `CHATQEC_MCP_SERVER` (and `--mcp-server`) take a comma-separated
list, so one variable can name both:

```bash
git clone https://github.com/QSCSoftwareEcosystem/QAppsWiki ~/QAppsWiki

CHATQEC_MCP_SERVER="chatqec-tools,sh -c 'PYTHONPATH=/path/to/QAppsWiki/tools python -m qappswiki serve --root /path/to/QAppsWiki'" \
  uv run streamlit run chatqec_app/streamlit_app.py
```

`MultiMCPClient` fans out to both servers and skips whichever fails to start,
so a broken wiki clone degrades to Stim-only tools rather than a hard failure.
The wiki's MCP server is a **stdio subprocess that reads its own working
tree** (there is no hosted endpoint), so **QAppsWiki must be cloned locally**
and the paths above point at that clone. The `qappswiki` console script isn't
on `PATH` in this setup, hence launching it via `PYTHONPATH` + `python -m qappswiki` rather than a bare `qappswiki serve`.

Two knobs in `conf/config.yaml` control the wiki side once its MCP server is
reachable:

```yaml
wiki:
  enabled: true   # query the wiki on every question; false = corpus-only
  k: 8             # sections requested per query, before reranking
```

Wiki sections are reranked together with Qdrant chunks by the same
cross-encoder. Each one carries an authority tier (`wiki_curated` or
`wiki_unverified`) derived from the page's own `provenance_status` frontmatter,
but that tier is currently only a label passed into the model's context block,
not a ranking weight: `TIER_WEIGHTS` is read solely by the corpus candidate
scoring, which wiki sections bypass by being appended straight to the rerank
pool. Wiki content therefore wins or loses purely on cross-encoder score today.
Whether the tier should also move the ranking is a deliberately deferred design
decision. See
[`docs/DESIGN_CHOICES.md`](docs/DESIGN_CHOICES.md#why-qappswiki-search-lives-in-qappswiki)
for the rationale. Use `-v` on `chatqec ask`/`chat` to trace whether the wiki
was queried and what it returned (there's no `CHATQEC_TRACE` env var; tracing
is a CLI flag).

This is new and **not yet measured** against the eval set. The recommended
way to check whether the wiki actually helps, or crowds out better corpus
chunks, is the same 62-pair eval run already used for every other retrieval
change:

```bash
uv run chatqec eval run --output eval/results.json
```

compared once corpus-only and once with the wiki's MCP server named, reading
off recall@K and citation accuracy for each (see [Eval](#eval)). **That
comparison has not been run for this feature.** One caveat as of this
writing: `chatqec eval run` has no `--mcp-server` option and does not read
`CHATQEC_MCP_SERVER`, unlike `ask`/`chat`, so today it always evaluates the
corpus-only pipeline regardless of environment. Until eval wiring picks up an
MCP server the same way `ask`/`chat` do, spot-check individual questions
instead: `chatqec ask --mcp-server "chatqec-tools,sh -c '...'" "<question>"` against the same question without `--mcp-server`, comparing
citations.

### Run the Streamlit app

The app **starts a local Qdrant itself** if one isn't already listening, so
this is usually all you need:

```bash
CHATQEC_MCP_SERVER=chatqec-tools uv run streamlit run chatqec_app/streamlit_app.py
```

Autostart only fires when `store.qdrant_url` is local and a `qdrant` binary is
on PATH; it reuses `qdrant_data/`, logs to `logs/qdrant.log`, and leaves an
already-running Qdrant alone. Set `CHATQEC_AUTOSTART_QDRANT=0` to switch it off,
which hosted deploys should do. Without a binary you get the start command in
the error instead of a bare `Connection refused`.

To start Qdrant yourself (`docker compose up -d`, or the native binary):

```bash
mkdir -p logs
QDRANT__STORAGE__STORAGE_PATH="$(pwd)/qdrant_data/storage" \
  QDRANT__STORAGE__SNAPSHOTS_PATH="$(pwd)/qdrant_data/snapshots" \
  setsid nohup qdrant > logs/qdrant.log 2>&1 < /dev/null &
curl -s http://localhost:6333/collections   # expect {"result":{"collections":[{"name":"chatqec_chunks"}]},...}
```

Then:

```bash
uv run streamlit run chatqec_app/streamlit_app.py

# …or with QEC tool use enabled (same as `ask --mcp-server`):
CHATQEC_MCP_SERVER=chatqec-tools uv run streamlit run chatqec_app/streamlit_app.py
```

Four pages: **Chat** (streaming Q&A with citation cards), **Sources**
(filterable registry view), **Watchlist** (what we track), **About**. Set
`CHATQEC_MCP_SERVER` to let the Chat page run the QEC tools (off by default).

> **`CHATQEC_MCP_SERVER` must be set _before_ you launch Streamlit, and the app
> restarted** to pick it up — the bot is built once and cached (`@st.cache_resource`).
> Without it the Chat page has no tools, so the model answers from the corpus and
> says it *"couldn't invoke the live simulation tools."* Make sure the tools are
> installed/up to date first (see
> [Enable QEC tool execution](#enable-qec-tool-execution)
> above).

### CLI commands at a glance

```bash
chatqec ask "..."         # one-shot question with citations
chatqec chat              # interactive REPL (/quit, /clear)
chatqec ingest --arxiv 2408.13687 --url https://...   # add a source (abstract only)
chatqec ingest --arxiv 2408.13687 --full-text         # ingest the full paper (LaTeX source → Marker PDF fallback)
chatqec ingest --pdf Raw/book.pdf --pages 46-77 --title "..." --describe-figures  # book chapter + figure vision
chatqec ingest --github QuTone/LightStim --commit-sha <sha> --github-paths docs/api --github-paths skills  # pinned repo docs
chatqec embed --all-pending --provider gemini         # embed → flip to active
chatqec sources list                                  # registry view
chatqec retrieve --query "..."                        # debug: top-K chunks only (no LLM)
chatqec canonical validate                            # check the canonical wiki
chatqec canonical embed                               # chunk + embed canonical pages into Qdrant
chatqec ask "..." --image fig.png                     # multimodal: attach an image to the question
chatqec watch refresh --apply                         # pull new arXiv/GitHub candidates
chatqec eval run --output eval/results.json           # run the 62-pair eval set
```

> **Tip — verify retrieval without Anthropic credit.** `chatqec retrieve --query "..."` exercises only Gemini embeddings + Qdrant + the cross-encoder reranker, **no
> Claude calls**. Use it to confirm the corpus is queryable before spending on
> the LLM synthesis step.

### Full-text vs. abstract ingestion

By default `chatqec ingest --arxiv <id>` stores only the **abstract** (one
chunk) — cheap, high-signal coverage that's enough to *find and cite* a paper.
Add `--full-text` to ingest the whole paper for depth on *how/why* questions:

- **LaTeX source (primary).** arXiv ships the original LaTeX for most papers.
  Equations (`$...$`, `\begin{equation}`) and code (`lstlisting`, `verbatim`)
  are preserved verbatim; figure images are dropped but `\caption{}` text is
  kept. This is the highest-fidelity path and needs no extra dependencies.
- **Marker PDF fallback.** For the minority of papers arXiv serves as PDF-only,
  the PDF is extracted with [Marker](https://github.com/datalab-to/marker) run
  in an **isolated `uvx` environment** (it pins `transformers` 4.x; the main env
  keeps 5.x), reconstructing equations to LaTeX. Set `TORCH_DEVICE=cuda` to use
  a GPU. Models download on first run.
- **Abstract (last resort)** if neither path yields usable text.

Add `--describe-figures` to also harvest **figures** from a full-text arXiv
paper (**hybrid ingest**): the LaTeX chunks stay authoritative for text and
equations, while the paper's PDF is fetched purely to persist its figure images,
which are then vision-described and appended as searchable figure chunks — the
same figure pipeline as `--pdf` below. Requires the `vision` extra + a GPU; on a
host without one the harvest is skipped automatically and text ingestion is
unaffected. The arXiv publication **year is filled in automatically** from the
resolved paper metadata, so figure and text chunks share the same recency score.

```bash
chatqec ingest --arxiv 2408.13687 --full-text --describe-figures
```

Reserve `--full-text` for impactful papers — see
[`scripts/reingest_arxiv_fulltext.sh`](scripts/reingest_arxiv_fulltext.sh) for a
reproducible batch re-ingest of the foundational QEC set.

### PDF ingestion with figures (`chatqec ingest --pdf`)

`chatqec ingest --pdf` extracts a book chapter or paper PDF via Marker and, by
default, also persists every figure it finds to `data/figures/<source_id>/`,
describes each one with a **local Llama-3.2-11B-Vision-Instruct** model
(US-origin, runs on your own GPU — zero API cost), and injects the
description into the chunk text as a `> **Figure (auto-described):** …`
callout so figures become semantically searchable and get cited like any
other evidence. The figure path is recorded in the chunk's metadata and
surfaced back to the user at answer time (the CLI prints a `Figures:` block;
Streamlit renders the images inline).

```bash
# --year drives recency scoring (see below); --describe-figures is the default
# (pass --no-describe-figures to skip figure vision).
chatqec ingest --pdf Raw/book.pdf --pages 46-77 \
  --title "Introduction to QEC" --year 2013 --describe-figures
```

> **Always pass `--year <publication-year>` for `--pdf` ingests.** Retrieval
> scores each chunk partly by recency (a 5-year time decay), and figures inherit
> their source's year. arXiv ingests fill the year in automatically, but a PDF
> has no reliable year metadata — omit `--year` and the whole source (text *and*
> figures) is scored **neutral** (0.5) instead of by its real age. If you forgot
> it, you can patch `sources/registry.jsonl` and the store payload after the fact
> (`meta_year`), but setting it at ingest time is the clean path.

Vision description requires the optional `vision` extra and a one-time
Hugging Face login (Llama-3.2-Vision weights are **gated** — accept the
license on the model page first):

```bash
uv sync --extra vision                # or: pip install -e '.[vision]'
huggingface-cli login                  # needed once, before first use
```

Without the extra installed (or with `--no-describe-figures`), PDF ingestion
still works exactly as before — figures just won't be described or made
searchable. `data/figures/` is a build artifact and is gitignored, same as
`data/chunks/`.

### Web-search fallback (Tavily)

When a question isn't covered by the corpus or canonical wiki, ChatQEC can
fall back to a web search (Tavily) and answer from cited web results instead
of refusing. **Off by default** — enable it with `TAVILY_API_KEY` in `.env`
plus `CHATQEC_WEB_FALLBACK=true`. The trigger is deterministic: it fires only
when the best retrieved chunk's cross-encoder rerank score falls below a
calibrated threshold (0.4). Web results sit at the lowest `web` authority tier
(0.3, below `blog`), so they never outrank canonical pages or papers, and the
model discloses when it relies on web sources and cites the URL. Results are
ephemeral — nothing is written to the corpus — and the CLI prints a
`chatqec ingest --url <url>` tip so you can promote a good source into the
permanent corpus. No new dependency: Tavily is called via the existing
`requests` library.

---

## Example usage

For a guided tour of every feature with real outputs, open
[`notebooks/getting_started.ipynb`](notebooks/getting_started.ipynb): each QEC
tool called directly (code parameters, threshold sweeps, LightStim codes,
decoder comparisons including NVIDIA GPU RelayBP on IBM's gross code, Tsim T
gates, GLCB links, IBM mapping), then a few `ChatQEC.ask` questions. Cells skip
cleanly when an extra, GPU, or account is missing. Re-run it as a test with
`uv run pytest --nbmake notebooks/` (needs Qdrant and an LLM provider; set
`CHATQEC_PROVIDER=local` with `scripts/serve_local_models.sh` to avoid API costs).

A few representative questions to try once the corpus is loaded (see
[Quick start](#quick-start) above):

```bash
# Definitional — always answered from the canonical wiki
uv run chatqec ask "What is the surface code and how does it detect errors?"

# Comparison / math-detail — pulls from arXiv papers in the corpus
uv run chatqec ask "How does the threshold theorem relate to the break-even point demonstrated on real hardware?"

# Needs a real simulator, not a guess — requires --mcp-server (see Enable QEC tool execution above)
uv run chatqec ask --mcp-server chatqec-tools \
  "Simulate a distance-3 surface code under depolarizing noise at p=0.001 for 5000 shots and report the logical error rate."

# Interactive REPL
uv run chatqec chat
```

The same questions work from the **Streamlit Chat page**
(`uv run streamlit run chatqec_app/streamlit_app.py`, with
`CHATQEC_MCP_SERVER=chatqec-tools` set first for the simulator example) —
answers stream in with clickable citation cards.

Every answer is a **cited answer or an honest refusal** — see [What to expect
on your first run](#what-to-expect-on-your-first-run) below for the topics
covered by the current seed corpus. For a much longer, continuously-verified
list of example queries (QAppsWiki-boosted lookups, LightStim/Tsim circuit
construction, PDF figure citations, corpus statistics, ...), see
[`docs/EXAMPLES.md`](docs/EXAMPLES.md).

---

## What to expect on your first run

**The corpus is intentionally small.** A fresh install indexes the 60 canonical
pages (surface code, stabilizer formalism, CSS codes, threshold theorem, magic
state distillation, and more; see [`knowledge/canonical/`](knowledge/canonical/)),
so definitional questions get cited answers and every tool works. The full
corpus (135 sources) adds arXiv papers, vendor pages, and textbook chapters;
load it from a snapshot or ingest sources yourself. Questions outside the
indexed corpus get an honest refusal, not a hallucinated answer.

**Costs.** A typical single-turn query costs roughly $0.01–$0.03 (one Haiku
classification + one Haiku rewrite + one Sonnet synthesis, all uncached).
Streaming and tool-use add a small overhead. The `chatqec.guards.CostBudget`
caps daily spend; the default is $5/day and is configurable via
`CHATQEC_DAILY_BUDGET_USD`.

**LLM provider is configurable and falls back automatically.** The default
config is `models.provider: "auto"` — calls go to Anthropic first; if your
Anthropic credit is empty (or the key is missing, rate-limited, or 403s) the
same call transparently falls through to Gemini, then to HuggingFace
open-weights (Llama-3.1-8B for classifier+rewriter, Llama-3.3-70B for
synthesizer+judge) if Gemini is also unavailable. Every call logs
which backend served it to stderr — there's no silent quality drop:

```
$ chatqec ask "..."
[chatqec] backend=anthropic unavailable (BadRequestError: credit balance too low) → trying next
[chatqec] backend=gemini unavailable (PermissionDenied: API key not valid) → trying next
[chatqec] backend=huggingface model=meta-llama/Llama-3.3-70B-Instruct
The surface code threshold under depolarizing noise is...
```

Five values are available for `models.provider`:

| Value                  | Behaviour                                                                                                                            |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `auto` *(default)* | Try Anthropic → Gemini → HuggingFace, falling back on credit/auth/quota errors. Non-recoverable errors (5xx, network) still raise. |
| `anthropic`          | Anthropic only. Fails hard if no credit / no key.                                                                                    |
| `gemini`             | Gemini only. Fails hard if no credit / no key.                                                                                       |
| `huggingface`        | HuggingFace only. Useful for fully open-stack deploys.                                                                               |
| `local`              | Self-hosted OpenAI-compatible servers (`models.local` in `conf/config.yaml`, e.g. vLLM); no API cost.                                |

Provide whichever keys you have: `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, and/or
`HF_TOKEN` (the
fine-grained token needs the **"Make calls to Inference Providers"** scope).
`auto` skips any backend whose key is missing — set just one and it works.

Tool use works with every provider except `huggingface`. Expect a quality
drop on synthesis when the fallback kicks in; pin one provider for evaluation
runs so results are not mixed across backends.

---

## What's in the box

| Layer            | Module                                                                            | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ---------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Canonical wiki   | [`knowledge/canonical/`](knowledge/canonical/)                                   | 60 pages (full v1 roster + v0.4 additions in[`_roster.yaml`](knowledge/canonical/_roster.yaml)); chunked + embedded via `chatqec canonical embed`                                                                                                                                                                                                                                                                                                                                                                                           |
| Source registry  | [`sources/registry.jsonl`](sources/registry.jsonl)                               | Typed JSONL; lifecycle states (`active`/`superseded`/`retracted`/`stale`/`pending_review`)                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Watchlist        | [`sources/watchlist.yaml`](sources/watchlist.yaml)                               | arXiv categories, GitHub repos, YouTube channels                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Ingestion        | [`src/chatqec/ingestion/`](src/chatqec/ingestion/)                               | Fetchers (arXiv abstract / arXiv full-text LaTeX source / Marker PDF / GitHub, pinned to an exact commit via`--commit-sha` / YouTube / URL), math-aware chunker, title-and-author resolver, local Llama-3.2-Vision figure describer (`--extra vision`)                                                                                                                                                                                                                                                                                     |
| Embedding        | [`src/chatqec/embedding/`](src/chatqec/embedding/)                               | Gemini `gemini-embedding-001` (default) or local Nomic; BM25 via fastembed                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Store            | [`src/chatqec/store/qdrant.py`](src/chatqec/store/qdrant.py)                     | Hybrid dense + sparse search                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Retrieval        | [`src/chatqec/retrieval/`](src/chatqec/retrieval/)                               | Spec §8.5 weighted scoring (`0.5·dense + 0.3·bm25 + 0.15·tier + 0.05·recency`), `mxbai-rerank-base-v1` cross-encoder reranker, canonical force-injection                                                                                                                                                                                                                                                                                                                                                                                                   |
| Conversation     | [`src/chatqec/conversation/`](src/chatqec/conversation/)                         | Haiku off-topic classifier + query rewriter; Sonnet synthesizer with native tool-use loop                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Guards           | [`src/chatqec/guards/`](src/chatqec/guards/)                                     | Per-IP rate limit, daily USD budget, hCaptcha verifier                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Eval             | [`src/chatqec/eval/`](src/chatqec/eval/) + [`eval/`](eval/)                     | 62-pair Q&A set across 7 subtypes; recall@K + citation accuracy + LLM-judge faithfulness                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| QEC tools        | [`packages/chatqec-tools`](packages/chatqec-tools) | 28 tools in four groups (see [Enable QEC tool execution](#enable-qec-tool-execution)); `chatqec-tools` runs one process per call, `chatqec-mcp-tools` serves the same tools over stdio MCP |
| QAppsWiki search | [`src/chatqec/retrieval/wiki_search.py`](src/chatqec/retrieval/wiki_search.py)   | Wraps the[QAppsWiki](https://github.com/QSCSoftwareEcosystem/QAppsWiki) MCP `search_pages` tool (self-hosted BM25 over the sections of 780 concept pages) into `RetrievedChunk`s that join the same cross-encoder rerank pass as corpus chunks; needs the wiki's MCP server named alongside `chatqec-tools` (see [Enable QAppsWiki search](#enable-qappswiki-search))                                                                                                                                                             |

The full corpus has 3,266 chunks across 135 sources (60 canonical, 41 arXiv,
26 blog/vendor, 6 book chapters, 2 pinned GitHub-doc sources), as recorded in
`sources/registry.jsonl`; `chatqec corpus-stats` recomputes the breakdown.

---

## How it works

![ChatQEC architecture: ingestion lane, shared corpus & tools, query lane, and a planned hardware-aware placement lane](docs/images/chatqec_architecture.png)

*Ingestion (offline), the shared corpus/tools layer, and the query pipeline are
implemented. The bottom **hardware** lane, drawn as future work, now exists as
the IBM Quantum and IQM tools (live calibration, circuit mapping, preview,
submission); the diagram predates them.*

The query lane in detail:

```
                       ┌──────────────┐
question  ──▶ Haiku ──▶│  off-topic?  │── no ──▶ refuse politely
                       └──────┬───────┘
                              │ yes
                       ┌──────▼───────┐
                       │   Haiku      │  resolves pronouns,
                       │   rewriter   │  emits 2–3 search variants
                       └──────┬───────┘
                              │
        ┌─────────────────────▼──────────────────────┐
        │      Hybrid retriever (Qdrant + BM25)      │
        │ → spec §8.5 scoring → mxbai reranker → top-K│
        │     ▲                                       │
        │     │  canonical wiki force-inject          │
        └─────┼───────────────────────────────────────┘
              │
       ┌──────▼─────────┐         ┌──────────────────────────┐
       │ Sonnet         │  tools  │  chatqec-tools (28)      │
       │ synthesizer    │ ◀────▶ │  one process per call:   │
       │ (cited answer) │         │  Stim, decoders, FT      │
       └────────────────┘         │  compile, QEC Studio,    │
                                   │  IBM Quantum, IQM        │
                                   └──────────────────────────┘
```

> When retrieval is weak (best rerank score < 0.4) and the web fallback is
> enabled, Tavily results are injected into the hybrid retriever as
> lowest-authority `web`-tier context instead of the pipeline refusing.

**Why each piece?** See [`docs/DESIGN_CHOICES.md`](docs/DESIGN_CHOICES.md) for a
focused walkthrough of model selection (why Haiku for the classifier, Sonnet
for the synthesizer, the cross-encoder reranker, etc.) and architectural picks (why
hybrid retrieval, why Qdrant, why MCP for tool use, why a canonical wiki
layer, why auto-fallback). Full design spec:
[`docs/planning/specs/2026-05-29-chatqec-design.md`](docs/planning/specs/2026-05-29-chatqec-design.md).

---

## Configuration

ChatQEC uses Hydra + OmegaConf. The root config is
[`conf/config.yaml`](conf/config.yaml); override anything from the command line:

```bash
uv run chatqec ask "..." --mcp-server chatqec-tools
CHATQEC_DAILY_BUDGET_USD=2.50 uv run streamlit run chatqec_app/streamlit_app.py
```

Required environment:

| Variable                                    | Purpose                                                                                                            |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `GEMINI_API_KEY`                          | Required for retrieval: `gemini-embedding-001` embeds every question (or set `embedding.provider=local` and rebuild the index); also answers when no Anthropic key is set |
| `ANTHROPIC_API_KEY`                       | Optional. Haiku (classifier/rewriter) + Sonnet (synthesizer + judge); used first under `models.provider=auto` (default) |
| `HF_TOKEN` *or* `HUGGINGFACE_API_KEY` | Llama-3.1/3.3 via HuggingFace Inference API — required when`models.provider=huggingface`                        |
| `HCAPTCHA_SECRET`                         | Optional. Public Streamlit deploys; empty = captcha disabled                                                       |
| `CHATQEC_DAILY_BUDGET_USD`                | Optional. Daily USD cap for the budget guard                                                                       |
| `CHATQEC_MCP_SERVER`                      | Optional. Tool commands for the web app, e.g. `chatqec-tools` (comma-separate to add the QAppsWiki server)        |
| `CHATQEC_PROVIDER`                        | Optional. Overrides `models.provider` (e.g. `local`)                                                               |
| `QISKIT_IBM_TOKEN`, `QISKIT_IBM_INSTANCE` | Optional. IBM Quantum tools                                                                                        |
| `IQM_TOKEN`, `IQM_SERVER_URL`             | Optional. IQM tools                                                                                                |
| `TAVILY_API_KEY`                          | Optional. Web-search fallback (with `CHATQEC_WEB_FALLBACK=1`)                                                      |
| `WATCHLIST_PUSH_TOKEN`                    | CI-only. GitLab token for the scheduled watchlist refresh MR                                                       |

---

## Eval

Three evaluations ship with the repository; [`docs/benchmarks/REPRODUCE.md`](docs/benchmarks/REPRODUCE.md)
gives the exact commands, pins, and environment.

- **Tool use on external benchmarks** (`chatqec eval bench`): 40 decoder
  comparisons from [decoder-bench](https://github.com/satvikmaurya/decoder-bench)
  and 17 items from [FTPrimitiveBench](https://github.com/ShuwenKan/FTPrimitiveBench),
  scored for tool routing, exact arguments, numeric agreement with the
  benchmark's ground truth, and decoder ranking. Includes an audit that re-runs
  decoder-bench's own pipeline where its published values disagree.
- **ChatQEC versus the bare model** (`eval/compare/`): 27 items (literature,
  computation, definitions, false premises) answered with and without
  retrieval and tools.
- **Retrieval** (`chatqec eval run --cases eval/qa_set.jsonl`): recall@K,
  citation accuracy, and LLM-judged faithfulness over a 62-question set.

Result files in [`eval/results/`](eval/results/) record the commits,
benchmark pins, model, and tool versions they were produced with.

---

## Roadmap

**Done:**

- End-to-end pipeline: classifier → rewriter → hybrid retrieval → synthesizer, with canonical-page injection
- 60-page canonical wiki; PDF, arXiv full-text, GitHub, and figure ingestion; watchlist refresh
- 28 QEC tools in [`packages/chatqec-tools`](packages/chatqec-tools), run one process per call (also an MCP server)
- Decoder comparison across twelve decoders, FTPrimitiveBench and decoder-bench circuits, QuEra QEC Studio
- IBM Quantum and IQM hardware tools: calibration, mapping, preview, submission
- Benchmark evaluation with provenance-stamped results; Streamlit app with rate limit, budget, hCaptcha
- One-script install, `chatqec bootstrap`, and GitHub Codespaces

**Open:**

- A public corpus snapshot with per-source licences (third-party full text is not redistributed today)
- Retrieval evaluation numbers on the 62-question set
- A hosted demo with bring-your-own-key
- Video and lecture transcripts in the corpus (the fetcher exists)
- OS-level isolation for `guppy_run`

---

## Contributing

**Propose a source** for the corpus: open an issue using the
[`add_source` template](.gitlab/issue_templates/add_source.md). Maintainer
reviews license + relevance, then `chatqec ingest` plus the weekly watchlist
refresh keeps it current.

**Work on the code:**

```bash
uv sync --extra dev
uv run pytest                          # app tests
(cd packages/chatqec-tools && uv run --all-extras python -m pytest)   # tool tests
uv run ruff check src tests chatqec_app
uv run mypy src
uv run chatqec canonical validate
```

Conventional Commits + small, reviewable diffs. The design spec
([`docs/planning/specs/`](docs/planning/specs/)) and implementation plan
([`docs/planning/plans/`](docs/planning/plans/)) are the source of
truth — extend them before adding a feature outside the v1 scope.

---

## Citing

If ChatQEC helps your research, please cite it as:

```bibtex
@software{chatqec_2026,
  author  = {Afrose, Sharmin and Leyton-Ortega, Vicente A. and Seefried, Ethan and
             Stapleton, Nicholas and Deshpande, Sangram and Humble, Travis S. and
             Ghosal, Tirthankar},
  title   = {ChatQEC: an open-source research assistant for Quantum Error Correction},
  version = {0.1.0},
  year    = {2026},
  url     = {https://github.com/QSCSoftwareEcosystem/ChatQEC}
}
```

---

## Acknowledgements

This material is based upon work supported by the U.S. Department of Energy,
Office of Science, National Quantum Information Science Research Centers,
Quantum Science Center.
