# ChatQEC LightStim capability check — `qec_circuit_build`

**Status: 4 of 20 prompts run live** (2026-07-28), raw logs in
[`lightstim_capability_check_raw_logs/`](lightstim_capability_check_raw_logs/).
The full prompt set — including the 16 not yet spot-checked here — lives in
[`eval/lightstim_qec_prompts.json`](../eval/lightstim_qec_prompts.json) for
later benchmarking runs. Same format as
[`stim_getting_started_capability_check.md`](stim_getting_started_capability_check.md).

**Purpose.** Exercise `qec_circuit_build` (merged into `chatqec-mcp-tools`
main, backed by LightStim pinned to commit
`47dd74976780192ae901d91958863cced1165354`) for code families Stim doesn't
ship built-in — toric code, BB codes, PQRM — plus LightStim's own
construction of the 6.6.6 color code, and the newly-ingested LightStim corpus
(`arxiv:2604.21472` full-text paper + `github:QuTone/LightStim@47dd7497`
curated docs/skills).

## What was actually run

```bash
uv run chatqec ask --mcp-server chatqec-mcp-tools -v \
  --log-file docs/lightstim_capability_check_raw_logs/<id>.txt "<question>"
```

Model: `claude-sonnet-4-6` (synthesis, provider `anthropic`). Run 2026-07-28.

| ID (`eval/lightstim_qec_prompts.json`) | Prompt topic | Tool(s) actually called | Verdict |
|---|---|---|---|
| `ls-tool-001` | Toric code build + params | `qec_circuit_build(toric_code, d=3, r=3)`, `stim_diagram` | ✅ Correct — 56 detectors, 2 observables, matches reference exactly; also diagrammed the resulting circuit (composability with `stim_diagram` works, confirming the earlier output-size-cap fix didn't break this path at this size) |
| `ls-tool-002` | BB code build + qubit/observable count | `qec_circuit_build(bb_code, d=6, r=6)` | ✅ Correct — 144 qubits (72 data + 72 ancilla), 12 observables, 444 detectors, all matching reference; correctly explained the $[[72,12,6]]$ structure |
| `ls-cmp-001` | Stim `code_params` vs LightStim `qec_circuit_build` comparison | `code_params(surface_code_rotated, d=3)`, `qec_circuit_build(toric_code, d=3, r=3)` | ✅ Correct, thorough — both tools called as expected; correctly attributed the 56-vs-24-detector gap to the toric code's periodic boundaries (18 stabilizers vs 8) and $k=2$ vs $k=1$, not physical-error-rate differences |
| `ls-edge-001` | Unsupported `bb_code` distance (7) | `qec_circuit_build(bb_code, d=6, r=6)` — **not** called with `d=7` | ⚠️ Passed differently than designed — the model recognized 7 isn't a real BB preset *without ever calling the tool with 7* (likely from the schema description's documented valid-distance set added in the final review's fix wave) and proactively substituted `d=6` with an explanation of all three presets. This means the tool's actual `ok=False` graceful-failure path for this case was not exercised end-to-end via chat in this run — worth a follow-up prompt that forces the invalid call (e.g. "call qec_circuit_build with distance 7 anyway and show me the raw error") if verifying that specific code path via chat matters. |

**Bottom line:** all 3 build/comparison prompts got real, correct numbers
from the actual pinned LightStim commit — no hand-rolled circuits, no
hallucinated qubit counts. Both cited sources retrieved from the corpus
(`arxiv:2604.21472`, plus Stim/QEC background). The one edge-case prompt
revealed the model is good enough at reading the tool's schema that it
avoids invalid calls rather than triggering the graceful-failure path —
informative, but means that path still needs a more forceful prompt to
verify live.

## Prerequisites to re-run this (or run the remaining 16 prompts)

1. **Install `chatqec-mcp-tools[lightstim]`** (now merged to that repo's
   `main`, no worktree needed):
   ```bash
   uv tool install --force "/home/pza/chatqec-mcp-tools[lightstim]"
   ```
   Confirm it registered:
   ```bash
   uv run --directory /home/pza/chatqec-mcp-tools python -c "from chatqec_mcp_tools.server import build_server; print([t.name for t in build_server().list_tools()])"
   ```
   (`chatqec-mcp-tools --help` does **not** work for this — the console
   script has no CLI argument parsing.)
2. **Pin gotcha found during this run:** `chatqec-mcp-tools`'s `pyproject.toml`
   originally had `mcp>=0.9` with no upper bound. `uv tool install` (no
   lockfile) resolved `mcp==2.0.0`, whose `Server` class dropped the
   `list_tools`/`call_tool` decorators `server.py`'s real stdio entry point
   uses — the installed command crashed with `AttributeError` on startup.
   Fixed by capping `mcp>=0.9,<2` (commit `83a49ef`). If you see `Connection
   closed` / `McpError` from `chatqec ask`, check `chatqec-mcp-tools`'s
   installed `mcp` version first.
3. **`GITHUB_PERSONAL_ACCESS_TOKEN`** and **`ANTHROPIC_API_KEY`** in `.env`
   (auto-loaded by the CLI). Note: the env var is
   `GITHUB_PERSONAL_ACCESS_TOKEN`, not `GITHUB_TOKEN` — the latter was a bug
   fixed in commit `3a78d41`.
4. **Ingestion already done** — no need to re-run unless the pin bumps:
   `github:QuTone/LightStim@47dd7497` (79 chunks, `docs/api/` + `skills/`)
   and `arxiv:2604.21472` (39 chunks, full LaTeX source) are both `active`
   in `sources/registry.jsonl` and embedded in Qdrant.
5. Run a prompt from `eval/lightstim_qec_prompts.json`, save its raw log
   under `docs/lightstim_capability_check_raw_logs/<id>.txt`, and add a row
   to the verdict table above.

## Known limitations (verified against the real adapter)

The tool has four distinct failure modes, and they don't all fail the same
way — worth knowing before reading `ok=False` as "the" failure signature:

- **Distance outside the tool's schema range `[3, 15]`, or `protocol` other
  than `"memory"`**: both are rejected by **pydantic validation** before the
  function body runs at all — a validation error, not an `ok=False` payload.
  (`protocol` is a `Literal["memory"]`, so any other value is a schema
  rejection, the same bucket as an out-of-range distance — not a
  tool-level `ok=False`.)
- **In-range distance with no matching preset** (`bb_code` outside
  `{6, 10, 12}`, `pqrm` outside `{3, 7}`): a clear `ok=False` + `error_message`
  from the tool, not a crash or hand-rolled fallback circuit — though see
  `ls-edge-001` above: a capable model may avoid triggering this path at all
  by reading the schema description first. The `bb_code` preset table in
  `lightstim_adapter.py` was itself trimmed down to the presets whose
  distance fits in `[3, 15]` — i.e. this is an arbitrary tool limit stacked
  on top of LightStim's real presets; larger real BB presets exist upstream
  and were excluded here.
- **Even distance for `color_code_666`**: also `ok=False` — the family
  requires odd distance and the adapter checks this explicitly.

`num_detectors` scales with `rounds`; `num_qubits` does not — e.g.
`toric_code` distance 3 gives 56/92/182 detectors at rounds=3/5/10
respectively, always 36 qubits. Reference values in
`eval/lightstim_qec_prompts.json` use `rounds = distance` unless stated
otherwise.
