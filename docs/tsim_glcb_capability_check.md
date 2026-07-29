# ChatQEC Tsim + GLCB capability check — `tsim_simulate` / `glcb_visualize_url`

**Status: 5 of 20 prompts run live** (2026-07-28), raw logs in
[`tsim_glcb_capability_check_raw_logs/`](tsim_glcb_capability_check_raw_logs/).
The full prompt set — including the 15 not yet spot-checked here — lives in
[`eval/tsim_prompts.json`](../eval/tsim_prompts.json) for later benchmarking
runs. Same format as
[`docs/lightstim_capability_check.md`](lightstim_capability_check.md).

**Purpose.** Exercise `tsim_simulate` (Tsim, a GPU-accelerated Stim-superset
simulator that adds non-Clifford gates via stabilizer rank decomposition —
merged into `chatqec-mcp-tools` main, backed by `bloqade-tsim` `0.1.5`, a
normal versioned PyPI dependency, no commit-pin) and `glcb_visualize_url`
(shareable QuEra Gemini Logical Circuit Builder links, pure URL-encoding, no
network call) against the newly-ingested Tsim corpus (`arxiv:2604.01059` full
paper + `github:QuEraComputing/tsim@21757967` curated docs + 15 mintlify
`url:` docs pages).

## What was actually run

```bash
uv run chatqec ask --mcp-server chatqec-mcp-tools -v \
  --log-file docs/tsim_glcb_capability_check_raw_logs/<id>.txt "<question>"
```

Model: `claude-sonnet-4-6` (synthesis, provider `anthropic`); off-topic
classifier `claude-haiku-4-5`. Run 2026-07-28.

| ID (`eval/tsim_prompts.json`) | Prompt topic | Tool(s) actually called | Verdict |
|---|---|---|---|
| `ts-nonclifford-001` | `T` gate on Tsim vs. plain Stim | `tsim_simulate(RX 0\nT 0\nH 0\nM 0\nDETECTOR rec[-1], shots=200)` | ✅ Correct — real stochastic mix of 0/1 in the 20-row preview (`[0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,0,1,0]`, `shape=(200,1)`), matching the reference's "genuine mix, non-deterministic"; correctly stated plain Stim has no `T` gate and would reject the circuit, without inventing an exact Stim error string |
| `ts-magic-002` | Magic-state prep step (real Tsim tutorial circuit) + DETECTOR | `tsim_simulate(R 0\nR_X(-0.3040867239846964) 0\nT_DAG 0\nDEPOLARIZE1(0.05) 0\nM 0\nDETECTOR rec[-1], shots=2000)` | ✅ Correct — `ok=True, shape=(2000,1)`, preview mix of 0/1 (8/20 ones in the raw log); correctly described this as one state-prep step from the tutorial, not the full protocol, and correctly flagged Tsim's raw (non-XOR) detector convention rather than assuming Stim semantics |
| `glcb-build-001` | Bell-pair GLCB link | *(first two attempts: none — see below)*, then `glcb_visualize_url(qubits=2, gates=[h@0→[0], cx@1→[0,1]])` | ✅ Correct on the 3rd attempt — returned `https://gemini-logical-circuit-builder.vercel.app/#circuit={version:1.1,qubits:2,gates:[{gate:h,column:0,targets:[0]},{gate:cx,column:1,targets:[0,1]}],sampler:clifft}`, a byte-for-byte match to the reference URL. **The first two attempts (exact prompt wording from the eval file, and a verbatim retry) were both rejected pre-tool-call by the Haiku off-topic classifier** as `off_topic` ("Build a GLCB visualization link for the Bell-pair circuit...") — see Known limitations below. A third attempt prefixed with "For a QEC stabilizer-code demo, ..." passed classification and got the correct result |
| `ts-edge-003` | Invalid gate name | *(1st live attempt: a QEC-prefixed paraphrase — rejected)*, then `tsim_simulate(FOO_BAR_NOT_A_GATE 0\nM 0, shots=10)` on the 2nd, more explicit rephrasing | ✅ Correct on the 2nd live attempt — `ok=False, error_message="tsim error: Gate not found: 'FOO_BAR_NOT_A_GATE'"`, exact match to the reference; the model reported the real failure honestly rather than a hand-rolled explanation, and correctly generalized this as a compile-time gate-validation error. **Separately, in a follow-up verification run made *after* this passing attempt (not part of the original live sequence), the eval file's exact literal wording — extracted programmatically with `json.load` and confirmed byte-for-byte (`==`) equal to the JSON field, not retyped — was also rejected by the off-topic classifier** (`ts-edge-003-literal.txt`); an earlier, less rigorous re-run attempt had let a shell-quoting artifact turn the field's literal `\n` into a real newline, invalidating that first "literal" claim, since fixed |
| `ts-combined-001` | Bell pair: simulate + visualize with the SAME circuit | `tsim_simulate(H 0\nCNOT 0 1\nM 0 1\nDETECTOR rec[-1] rec[-2], shots=2000)`, `glcb_visualize_url(qubits=2, gates=[h@0→[0], cx@1→[0,1]])` | ✅ Correct and consistent — detector preview all `[0]` across all 2000 shots (perfect noiseless ZZ correlation, matching the reference); GLCB URL byte-for-byte identical to the reference; the model kept the gate order (H then CX on qubits 0,1) consistent across both tool calls |

**Bottom line:** all 5 prompts eventually got real, correct numbers/URLs from
the actual installed `bloqade-tsim==0.1.5` and the real `glcb_visualize_url`
encoder — no hand-rolled samples, no invented URLs, no fabricated error
strings. The one unplanned finding: **for both `glcb-build-001` and
`ts-edge-003`, the eval file's exact literal wording was rejected by the
off-topic classifier before any tool was ever considered** — confirmed for
`glcb-build-001` by two live runs of its literal wording (both rejected).
For `ts-edge-003` this claim took two correction passes to actually verify:
the initial task pass cited a paraphrased attempt as if it were the literal
wording (wrong); a first "fix" re-ran what was *typed* as the literal
wording but let shell-quoting silently turn the JSON field's literal `\n`
(backslash + letter `n`) into a real newline character, so it still tested a
different string than the eval file actually contains (wrong again, one
level more subtle). The claim is now confirmed the third time, correctly:
the question was extracted with `json.load` (not retyped), passed through a
shell variable, sent to `chatqec ask`, and the resulting logged
`ask.question` value was checked for **exact (`==`) string equality**
against the JSON field before trusting the result — see
`ts-edge-003-literal.txt` and the verification transcript in the task
report. It, too, was rejected as `off_topic`. A milder QEC-flavored
paraphrase of `ts-edge-003` ("For a QEC test, simulate this gate sequence
with Tsim: ...") was *also* rejected; only a more explicit rephrasing naming
"quantum error correction circuit simulation" and "stabilizer circuit"
passed — and that passing rephrased run happened chronologically *before*
the literal-wording confirmation above, not after, since the literal check
was only added in a later fix round. This is a genuine, reproducible
classifier limitation, not a fluke, documented honestly below rather than
papered over with a "successful" run that never actually happened for that
exact wording.

## Prerequisites to re-run this (or run the remaining 15 prompts)

1. **Install `chatqec-mcp-tools[tsim]`** (from the `quera-tsim-glcb` worktree
   at the time of this check; merge to `main` pending):
   ```bash
   uv tool install --force "/home/pza/chatqec-mcp-tools/.worktrees/quera-tsim-glcb[tsim]"
   ```
   Confirm it registered (expect all 8 tools):
   ```bash
   uv run --directory /home/pza/chatqec-mcp-tools/.worktrees/quera-tsim-glcb python -c \
     "from chatqec_mcp_tools.server import build_server; print([t.name for t in build_server().list_tools()])"
   ```
   Without the `tsim` extra, `tsim_simulate` is still registered but returns
   `ok=False` for every call (`tsim not installed: ...`); `glcb_visualize_url`
   has no optional dependency and always works.
2. **`GITHUB_PERSONAL_ACCESS_TOKEN`** and **`ANTHROPIC_API_KEY`** in `.env`
   (auto-loaded by the CLI).
3. **Ingestion already done** — no need to re-run unless the pin bumps:
   `arxiv:2604.01059` (24 chunks, full LaTeX source), `github:QuEraComputing/tsim@21757967`
   (6 chunks, `docs/`), and 15 `url:` mintlify pages are all `active` in
   `sources/registry.jsonl` and embedded in Qdrant.
4. Run a prompt from `eval/tsim_prompts.json`, save its raw log under
   `docs/tsim_glcb_capability_check_raw_logs/<id>.txt`, and add a row to the
   verdict table above.

## Known limitations (verified against the real adapter and the real classifier)

- **The off-topic classifier can reject a legitimate GLCB/Tsim tool-use
  prompt if it doesn't read as QEC-flavored — even the eval file's own
  literal wording.** Both of the five run prompts that have a `tool_edge_case`
  or `glcb_build` category (`glcb-build-001`, `ts-edge-003`) had their exact,
  unmodified `eval/tsim_prompts.json` question text rejected pre-tool-call
  with `classifier.response = off_topic` from `claude-haiku-4-5`:
  - `glcb-build-001`: rejected twice on the literal wording
    (`glcb-build-001.txt`, `glcb-build-001-retry.txt`); a QEC-framed
    rephrasing ("For a QEC stabilizer-code demo, ...") then passed
    (`glcb-build-001-rephrased.txt`).
  - `ts-edge-003`: rejected on the literal wording, but only confirmed
    reliably on the third attempt at documenting this. The passing run
    (`tsim_simulate(...)` succeeding, see the verdict table above) came from
    a QEC-framed rephrasing typed during the original live session — the
    eval file's exact literal wording was never actually tried live at that
    time. A first correction extracted what was believed to be the literal
    wording, but a shell-quoting slip converted the JSON field's literal
    `\n` (backslash + `n`) into a real newline, so it still tested the wrong
    string. The current, verified re-run (`ts-edge-003-literal.txt`) reads
    the question directly with `json.load(open("eval/tsim_prompts.json"))`
    and passes it through a shell variable with no retyping; the logged
    `ask.question` value was then checked and confirmed **exactly equal**
    (Python `==`) to the JSON field before being trusted. Result: `off_topic`
    again — the same verdict as before, now on genuinely the right input. A
    milder QEC-flavored paraphrase ("For a QEC test, simulate this gate
    sequence with Tsim: ...") was *also* rejected (first entry in
    `ts-edge-003.txt`); only an explicit rephrasing naming "quantum error
    correction circuit simulation" and "stabilizer circuit" passed
    classification (second entry in the same file).

  This is a real behavior of the shipped classifier prompt, not a bug in
  either MCP tool — both tools worked correctly once the question was
  classified on-topic. Worth knowing when writing prompts for these two
  tools in particular: circuit-building/visualization/simulation requests
  that don't mention QEC, stabilizers, fault tolerance, or decoding by name
  are at real risk of being turned away before they reach the tool-use loop
  — even fairly QEC-adjacent phrasing (e.g. "For a QEC test, ...") isn't
  always enough.
- **`tsim_simulate` always calls `compile_detector_sampler()` internally.** A
  circuit with `M` but no `DETECTOR`/`OBSERVABLE_INCLUDE` returns
  `samples_preview` with shape `(shots, 0)` — it does **not** return raw
  measurement bits the way `stim_simulate` does for the identical circuit
  text. Not exercised as a live prompt in this run (both `ts-clifford-*`
  prompts that hit this quirk are in the remaining 15), but confirmed in the
  reference values computed directly against the real tool
  (`eval/tsim_prompts.json`'s description field) and worth flagging for
  anyone reading `ok=True` with an empty preview as a failure.
- **Tsim's detector/observable convention is inverted from Stim's by
  default.** `use_reference_sample=False` (the tool's default) reports the
  *actual* detector outcome; Stim always reports a flip relative to a
  noiseless reference. `ts-combined-001` above exercised the deterministic,
  noiseless case (both conventions agree at `0`), so this run alone doesn't
  distinguish the two — see `ts-refconv-001`/`ts-refconv-002` in the
  remaining prompt set for a case where they actually diverge (`X 0\nM 0\nDETECTOR
  rec[-1]`: default reports `True` every shot, `use_reference_sample=true`
  reports `False` every shot).
- **Distinct failure modes, not exercised together here:** `tsim_simulate`
  rejects circuits over 256 KB (`_MAX_CIRCUIT_BYTES`) and requests over
  100,000 shots (`_MAX_SHOTS`, lower than `stim_simulate`'s 1,000,000 cap
  because Tsim's ZX/JAX pipeline is far more expensive per shot) with a clear
  `ok=False` before ever importing `tsim` — verified by reading
  `tools/tsim_simulate.py` directly, not run live in this check (`ts-edge-001`,
  `ts-edge-002` remain in the un-run 15). The invalid-gate case actually run
  here (`ts-edge-003`) fails one level deeper, inside `tsim.Circuit(...)`
  parsing, and surfaces the library's own exception text verbatim
  (`"tsim error: Gate not found: '...'"`)  rather than a tool-authored
  message — two different code paths for two different kinds of bad input.
- **`glcb_visualize_url` does no gate-name validation.** It is a pure
  string/URL encoder with no network call and no dependency on `bloqade-tsim`
  or `stim` — it will happily encode a `T` gate into a link even though
  that's exactly the gate plain Stim rejects, because the two tools don't
  cross-check each other's vocabulary. Not exercised as a live prompt here
  (`ts-combined-002` in the remaining 15 covers this directly), but confirmed
  by reading `tools/glcb_visualize.py`.
