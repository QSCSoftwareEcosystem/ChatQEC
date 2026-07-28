# ChatQEC LightStim capability check — `qec_circuit_build`

**Status: NOT YET RUN.** This environment has no `ANTHROPIC_API_KEY`, so
`chatqec ask` cannot make a real model call. This doc ships the prompts,
the exact runbook, and the pass criteria so whoever has credentials can run
it in ~5 minutes and fill in the verdict table — nothing below is a
fabricated result.

**Purpose.** 3 prompts that exercise `qec_circuit_build` (added in
`chatqec-mcp-tools`, branch `lightstim-integration`, backed by LightStim
pinned to commit `47dd74976780192ae901d91958863cced1165354`) for code
families Stim doesn't ship built-in: toric code, color code, BB codes,
PQRM. Same format as
[`stim_getting_started_capability_check.md`](stim_getting_started_capability_check.md).

## Prerequisites to actually run this

1. **Install the branch's MCP server build** (it's not merged/released yet,
   so the plain `chatqec-mcp-tools` console command won't have the new tool
   until this is done):
   ```bash
   uv tool install --force /home/pza/chatqec-mcp-tools/.worktrees/lightstim-integration
   ```
   Confirm it registered: `chatqec-mcp-tools --help` should now expose
   `qec_circuit_build` alongside the five existing tools (run
   `uv run --directory ~/chatqec-mcp-tools/.worktrees/lightstim-integration python -c "from chatqec_mcp_tools.server import build_server; print([t.name for t in build_server().list_tools()])"`
   to check without a global install, if you'd rather not install globally).
2. **Set `ANTHROPIC_API_KEY`** (or configure whichever provider `chatqec`
   is set up to use — `chatqec ask --mcp-server ...` requires the
   `anthropic`, `gemini`, or `auto` provider per `src/chatqec/cli.py`).
3. Run each prompt below and record the tool(s) actually called and the
   verdict, exactly as the Stim capability-check doc does. Save raw stdout
   per prompt under `docs/lightstim_capability_check_raw_logs/` (create it
   at that point — nothing to log yet, so it doesn't exist in this commit).

## Prompts

```bash
uv run chatqec ask --mcp-server chatqec-mcp-tools \
  "Build a distance-3 toric code memory circuit and tell me its [[n,k,d]] and detector count."

uv run chatqec ask --mcp-server chatqec-mcp-tools \
  "I need a distance-5 color code circuit for 5 rounds — can you build it and diagram it?"

uv run chatqec ask --mcp-server chatqec-mcp-tools \
  "Compare a distance-3 surface code (Stim) against a distance-3 toric code (LightStim) — same physical error rate, which has more detectors?"
```

## Pass criteria

`qec_circuit_build` gets called (not a hand-rolled circuit) for any family
only LightStim supports (toric/color/BB/PQRM), and the returned
`num_qubits`/`num_detectors` match the family's known values at that
distance **and rounds** (`num_detectors` scales with `rounds`; `num_qubits`
does not — e.g. `toric_code` distance 3 gives 56/92/182 detectors at
rounds=3/5/10 respectively, always 36 qubits). All reference values below
use **`rounds = distance`**, matching prompt #1's implicit assumption —
re-derive if you run a different rounds count.

Reference values (computed directly from `qec_circuit_build` against the
real pinned LightStim commit, in this worktree's venv — NOT model output,
use these to sanity-check whatever the model reports for prompts #1 and #2):

| Family | Distance | Rounds | num_qubits | num_detectors | num_observables |
|---|---|---|---|---|---|
| `toric_code` | 3 | 3 | 36 | 56 | 2 |
| `color_code_666` | 3 | 3 | 13 | 18 | 1 |
| `color_code_666` | 5 | 5 | 37 | 90 | 1 |
| `bb_code` | 6 | 6 | 144 | 444 | 12 |
| `pqrm` | 3 | 3 | 25 | 40 | 1 |

## Verdict table (fill in after running)

| # | Prompt topic | Tool(s) actually called | Verdict |
|---|---|---|---|
| 1 | Toric code build + params | _not run_ | _pending_ |
| 2 | Color code build + diagram | _not run_ | _pending_ |
| 3 | Stim vs LightStim comparison | _not run_ | _pending_ |

## Known limitations to watch for while running

- `bb_code` only supports distances `{6, 10, 12}` and `pqrm` only `{3, 7}`
  by construction (LightStim's real presets, not an arbitrary tool limit) —
  a prompt asking for an unsupported distance should get a clear
  `ok=False` error message back from the tool, not a crash or a
  hand-rolled fallback circuit. Worth adding as a 4th prompt if the first
  three pass cleanly.
- The knowledge-base ingestion of LightStim's docs (`chatqec ingest
  --github QuTone/LightStim --commit-sha 47dd74976780192ae901d91958863cced1165354`)
  has also not been run yet in this environment (no `GITHUB_TOKEN`) — so
  the model has no retrieved LightStim documentation to draw on for these
  prompts yet, only the tool's own input schema/description. That's a
  fair test of the tool in isolation, but re-run after ingestion lands to
  see if retrieved docs change the answers' quality.
