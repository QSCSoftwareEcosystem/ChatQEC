# Growing the corpus

How to add sources to the ChatQEC corpus beyond the canonical wiki that
`chatqec bootstrap` indexes. Every new source lands as `pending_review` in
`sources/registry.jsonl`; `chatqec embed --all-pending` embeds it and marks it
active. Moved here from the README.

## Full-text vs. abstract ingestion

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
[`scripts/reingest_arxiv_fulltext.sh`](../scripts/reingest_arxiv_fulltext.sh) for a
reproducible batch re-ingest of the foundational QEC set.

## PDF ingestion with figures (`chatqec ingest --pdf`)

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

## Web-search fallback (Tavily)

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
