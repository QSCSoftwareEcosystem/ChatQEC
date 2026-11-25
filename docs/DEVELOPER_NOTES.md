# Developer notes

Details for people changing ChatQEC or running it on a shared GPU machine.
Moved here from the README.

## Reinstalling the tools after a change

**The installed command is a snapshot.** `uv tool install` freezes the tools at
install time, so **after adding or changing a tool under `packages/chatqec-tools` you
must reinstall** (with the extras you use) or the change won't be visible:

```bash
uv tool install --force "./packages/chatqec-tools[ibm,decoders]"
```

Check what tools the model can actually see:

```bash
uv run python -c "from chatqec.cli import _build_chatqec; \
  b=_build_chatqec(mcp_server='chatqec-tools'); \
  print([t.name for t in b.synthesizer.mcp_adapter.client.list_tools()])"
# → 28 names: stim_simulate, stim_diagram, ..., ibm_*, iqm_*, qec_studio_*
```

## Optional tool extras: LightStim and Tsim

**`qec_circuit_build` needs the optional `lightstim` extra.**
The tool is always registered, but without `lightstim` installed it returns
`ok=False` for every call. Reinstall with the extra to actually use it:
`uv tool install --force "./packages/chatqec-tools[lightstim]"`. It builds circuits
for code families Stim doesn't ship built-in (toric code, BB codes, PQRM,
plus LightStim's own construction of the 6.6.6 color code) — see
[`docs/EXAMPLES.md`](EXAMPLES.md#lightstim-via-qec_circuit_build) and
[`docs/lightstim_capability_check.md`](lightstim_capability_check.md).

**`tsim_simulate` needs the optional `tsim` extra.** Like
`qec_circuit_build`, the tool is always registered but returns `ok=False`
for every call without it. Reinstall with the extra to actually use it:
`uv tool install --force "./packages/chatqec-tools[tsim]"`. Unlike `lightstim`,
`tsim` is a normal versioned PyPI dependency (`bloqade-tsim>=0.1.5,<0.2`,
no commit-pin needed — see
[`docs/DESIGN_CHOICES.md`](DESIGN_CHOICES.md#why-tsim-needs-no-commit-pin-for-its-code-dependency-only-for-kb-ingestion)).
It runs Stim-superset circuits — including non-Clifford gates (`T`,
`T_DAG`, `R_X`/`R_Y`/`R_Z`, ...) plain Stim rejects outright — via
[Tsim](https://github.com/QuEraComputing/tsim). `glcb_visualize_url` (no
extra needed) builds a shareable QuEra Gemini Studio link
for a small circuit. See
[`docs/EXAMPLES.md`](EXAMPLES.md#tsim--glcb-via-tsim_simulate--glcb_visualize_url)
and [`docs/tsim_glcb_capability_check.md`](tsim_glcb_capability_check.md).

## Python versions and local GPU models

**Python version.** `uv tool install` picks your newest Python (3.13 on a current machine), and
guppylang 1.x needs 3.12 or newer, so you get different package versions from the 3.11 development
environment (guppylang 1.2 against 0.21; in 1.x a measurement must be read with
`measure(q).read()`). The tool suite is verified on both Python 3.11 and 3.13.

**Running next to local vLLM models.** Two things break the app on a shared GPU box.
(1) An expired `HF_TOKEN` exported in your shell (for example in `~/.bashrc`) is not overridden by
`.env`, and `huggingface_hub` then answers 401 even for the public, cached reranker; refresh the
token or start the app with `HF_HUB_OFFLINE=1`. (2) The reranker and the CUDA-Q decoders default
to GPU 0, which a served model can fill (`CUDA out of memory`, and CUDA-Q prints its error on
stdout); pin them to a free GPU with `CUDA_VISIBLE_DEVICES=<n>`.

## QAppsWiki internals

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
[`docs/DESIGN_CHOICES.md`](DESIGN_CHOICES.md#why-qappswiki-search-lives-in-qappswiki)
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
off recall@K and citation accuracy for each (see [Eval](../README.md#eval)). **That
comparison has not been run for this feature.** One caveat as of this
writing: `chatqec eval run` has no `--mcp-server` option and does not read
`CHATQEC_MCP_SERVER`, unlike `ask`/`chat`, so today it always evaluates the
corpus-only pipeline regardless of environment. Until eval wiring picks up an
MCP server the same way `ask`/`chat` do, spot-check individual questions
instead: `chatqec ask --mcp-server "chatqec-tools,sh -c '...'" "<question>"` against the same question without `--mcp-server`, comparing
citations.
