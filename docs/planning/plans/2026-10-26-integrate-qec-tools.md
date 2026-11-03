# Integrate the vendor QEC tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the vendor QEC tools (CUDA-Q QEC, Tesseract, bloqade-decoders, Qiskit Paulice, Qiskit FT transpile, Guppy/Selene, Deltakit, BB5/tile code parameters) callable from chatQEC chat, with the two user flows (decoder comparison, IBM Pauli checks) working end to end.

**Architecture:** Tools are MCP tools in the sibling repo `~/chatqec-mcp-tools` (one pydantic Input/Output pair per tool in `src/chatqec_mcp_tools/tools/`, registered as a `ToolSpec` in `server.py`, heavy deps behind optional extras, imported lazily). chatQEC itself only gets prompt text, UI rendering, tests, RAG sources and notebook sections.

**Tech Stack:** Python 3.11, uv, pydantic, Stim, pytest, ruff, mypy; vendor SDKs per task; local vLLM models (Muse-Glimmer :8001, Gemma 4 :8002) for all chat testing.

**Spec:** `docs/planning/specs/2026-10-26-integrate-qec-tools-design.md`

## Global Constraints

- Work on `main` in both repos; never create branches. Commit after each finished task, one commit per repo, Conventional Commits (`feat(tools): ...`).
- **No `Co-Authored-By: Claude` trailer** and no "Generated with Claude Code" line in commits (user rule, overrides the default attribution).
- Testing through chat uses local models only (`CHATQEC_PROVIDER=local`); no Claude or Gemini tokens. Gemini is allowed only for embedding into Qdrant (the collection's vector space).
- Deltakit: local decoders only, nothing that costs money. Cloud decoders report "not configured".
- Heavy vendor imports are lazy (inside functions). `server.py` must import cleanly with no extras installed; a missing extra returns `ok=False` with `uv tool install 'chatqec-mcp-tools[<extra>]'` in the message.
- New tool outputs are pydantic models with `ok: bool` and `error_message: str`, `ConfigDict(extra="forbid")`, matching existing tools.
- BB5, barbell, tile numbers are copied from the ingested papers with a source id, never computed or recalled.
- Stage only the files each commit touches. chatQEC has unrelated uncommitted eval work (`api.py`, `cli.py`, `eval/*`, `providers/gemini.py`, their tests, `sources/registry.jsonl`) that must not enter these commits. No stash.
- Python 3.11; files stay under ~400 lines (split helpers into `_<name>.py` as the existing tools do).
- No em-dashes in prose written for the user (docs, notebook markdown, commit bodies).

## Review Focus

- Circuit with no noise or no observables passed to a new tool: clear `ok=False` message, not a crash (pattern: `decoder_compare` lines 220-245).
- Non-Clifford or measurement-heavy circuit given to a Stim-based path (Pauli checks, decoders): rejected with a message naming the cause.
- Oversized circuit or shot count: capped or rejected before the 60 s MCP budget, as `_MAX_CIRCUIT_BYTES` and `_MAX_SHOTS` do.
- Vendor native crash (segfault) in a decoder: the server survives (child process, as `_isolated_decode.py`).
- `guppy_run` given hostile Python source: runs only in the sandbox with limits, no network, no files outside a temp dir.
- Extra missing or no GPU: per-tool `ok=False` with an install hint, and the tool still appears in `list_tools`.

## Per-tool chatQEC wiring (referenced by Tasks 1-6, do not repeat)

For each new tool name `<tool>`:

1. `src/chatqec/conversation/prompts.py` (tool guidance near lines 32-46): one bullet saying when to call `<tool>`, with the same wording style as the `stim_simulate` and `code_params` bullets.
2. `chatqec_app/components/tool_output.py:65`: add `<tool>` to the `("pymatching_decode", "code_params")` tuple unless the output needs a table (then render `rows` like the threshold sweep branch above it).
3. `tests/test_prompts_tools.py`: add `<tool>` to the names tuple in the existing loop (line 7).
4. `notebooks/getting_started.ipynb`: one markdown cell plus one code cell using the notebook's `tool(name, **args)` helper and a new `CAN[...]` flag in the setup cell (cell 2) when the extra or hardware is needed; insert after the section that fits (decoders after cell 16, IBM after cell 22, new sections before "8. Asking ChatQEC"). Execute with `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/getting_started.ipynb` and keep outputs.
5. Chat check: reinstall the tool server (`uv tool install --force ~/chatqec-mcp-tools[<extras>]`), start the models (`scripts/serve_local_models.sh all`), then run `CHATQEC_PROVIDER=local CHATQEC_MCP_SERVER=chatqec-mcp-tools uv run chatqec ask "<prompt>"` and confirm the trace shows a `<tool>` call. If the classifier or router blocks the question, fix the prompt or routing, not the test. Task 7 repeats the check in the Streamlit UI.

Commit in `~/chatqec-mcp-tools` first (tool, tests, `pyproject.toml` extra), then in `~/chatQEC` (wiring, tests, notebook).

## File Structure

`~/chatqec-mcp-tools/src/chatqec_mcp_tools/tools/`:
- `decoder_compare.py`, `_isolated_decode.py`, `_cudaq_decoders.py`: modified (Task 1)
- `ibm_hardware/pauli_checks.py` + `_pauli_checks_impl.py`: new (Task 2)
- `ibm_hardware/ft_compile.py`: new (Task 3)
- `deltakit_memory.py`: new (Task 4)
- `guppy_run.py` + `_guppy_runner.py` (the script the sandbox executes): new (Task 5)
- `code_params.py` + `_paper_codes.py` (static table with sources): modified/new (Task 6)
- `server.py`: one import, one handler, one `ToolSpec` per new tool
- `pyproject.toml`: extras `paulice`, `ft`, `deltakit`, `guppy`; extend `decoders` and `cudaq` only if the API-discovery step shows new pins are needed

## Task 0: Baseline

**Files:** none changed.

- [ ] **Step 1:** In both repos run `uv run pytest -q` and `uv run ruff check .`. Record the pass counts. Expected: all green (chatQEC's uncommitted eval tests may fail; note them, do not fix).
- [ ] **Step 2:** `scripts/serve_local_models.sh all`; confirm `curl -s localhost:8001/v1/models` and `:8002` list `muse-glimmer` and `gemma-4`.
- [ ] **Step 3:** Run `CHATQEC_PROVIDER=local CHATQEC_MCP_SERVER=chatqec-mcp-tools uv run chatqec ask "Compare PyMatching and Tesseract on a distance-3 surface code at p=0.002"`. Expected: a `decoder_compare` call in the trace and an answer. This is the baseline for Task 1 and proves the local-model path works.

## Task 1: Widen `decoder_compare`

**Files:**
- Modify: `tools/decoder_compare.py` (`DecoderName`, `DecoderCompareInput`, `_run_one`), `tools/_isolated_decode.py` (`_build`), `tools/_cudaq_decoders.py` (`CUDAQ_DECODERS`, `cudaq_predict`)
- Test: `tests/test_decoder_compare.py`

**Interfaces:**
- Produces: `DecoderName` gains `"mwpf"`, `"mle"`, `"table"` (bloqade-decoders, `decoders` extra, child process) and `"cudaq_sliding_window"`, `"cudaq_ai_predecoder"` (`cudaq` extra, GPU). `DecoderCompareInput` gains `window_size: int | None = Field(None, ge=2)` and `window_step: int | None = Field(None, ge=1)`, used only by `cudaq_sliding_window`.
- Consumes: existing `predict_isolated(name, dem_text, det, timeout) -> (pred, seconds)` and `cudaq_predict(name, dem, det)`.

- [ ] **Step 1: API discovery.** In a scratch venv (`uv venv $SCRATCH/dc && uv pip install -p $SCRATCH/dc/bin/python 'bloqade-decoders>=0.6,<0.7' 'cudaq-qec-cu13>=0.8.1,<0.9'`), list what `bloqade.decoders` and `cudaq_qec` expose for MWPF, MLE, table, sliding window and the AI pre-decoder. Write the findings (class names, constructor args, what the pre-decoder needs, e.g. a model checkpoint) as a short comment block at the top of `_isolated_decode.py` / `_cudaq_decoders.py`. If a decoder does not exist in the installed version, drop it from `DecoderName` and say so in the commit body.
- [ ] **Step 2: Failing tests** in `tests/test_decoder_compare.py`: `test_new_decoder_names_accepted` (each new name validates in `DecoderCompareInput`), `test_window_params_rejected_without_sliding_window` (validator raises if `window_size` set and `cudaq_sliding_window` not in `decoders`), `test_missing_extra_reports_hint` (patch the import to raise `ImportError`, result has `ok=False` and `chatqec-mcp-tools[decoders]` or `[cudaq]` in `note`), and, skipped unless the extra is importable (`pytest.importorskip`), `test_mle_not_worse_than_pymatching_on_repetition_code` (d=3 repetition, p=0.01, `seed=1`, 2000 shots: `mle.logical_error_rate <= pymatching.logical_error_rate + 3 * pymatching.std_error`) and `test_mwpf_and_table_ok_on_surface_d3`.
- [ ] **Step 3:** Run `uv run pytest tests/test_decoder_compare.py -v`. Expected: the new tests FAIL.
- [ ] **Step 4: Implement.** Extend `DecoderName`; route `mwpf`, `mle`, `table` through `predict_isolated` (extend `_build`); route the two cudaq names through `cudaq_predict` (pass window params for sliding window); set `_run_one` notes for caps. Update the field description strings so the LLM sees every name and when to pick it. Add per-decoder shot caps next to `_MAX_TESSERACT_SHOTS` if discovery shows a slow decoder.
- [ ] **Step 5:** Install extras locally (`uv pip install -e '.[decoders,cudaq]'`), run the full test file, then `uv run ruff check . && uv run mypy`. Expected: PASS, clean. On this machine (2x RTX 6000 Ada) the cudaq tests run for real.
- [ ] **Step 6:** Wire chatQEC per "Per-tool chatQEC wiring" for `decoder_compare` (prompt bullet update only; the tool name already exists in the tuple and tests). Notebook: extend section 4 with a cell comparing `mwpf`, `mle`, `table` and one with sliding window on a multi-round circuit.
- [ ] **Step 7:** Chat check: "Compare PyMatching, Tesseract, MLE and the CUDA-Q sliding window decoder on a distance-5 surface code at p=0.002" against Muse then Gemma.
- [ ] **Step 8: Commit** in both repos (`feat(tools): add mwpf, mle, table and cudaq sliding-window/pre-decoder to decoder_compare`; `feat(chat): document the wider decoder set and extend the notebook`).

## Task 2: `ibm_add_pauli_checks` (Qiskit Paulice)

**Files:**
- Create: `tools/ibm_hardware/pauli_checks.py`, `tools/ibm_hardware/_pauli_checks_impl.py`
- Modify: `server.py`, `pyproject.toml` (extra `paulice = ["qiskit-paulice", "qiskit>=1.0"]`), `tests/test_base_install.py` (`_EXPECTED_TOOLS` gets the name; add it to the no-extra ok=False test list)
- Test: `tests/test_ibm_pauli_checks.py`

**Interfaces:**
- Produces:
  - `IbmAddPauliChecksInput`: `circuit: str`, `circuit_format: Literal["qasm", "stim"] = "qasm"`, `num_checks: int | None = Field(None, ge=1, le=64)`, `noise_p: float | None = Field(None, gt=0, le=0.1)`, `shots: int = Field(2000, ge=1, le=50_000)`, `seed: int | None = None`.
  - `IbmAddPauliChecksOutput`: `ok`, `circuit_qasm: str`, `overhead: PauliCheckOverhead`, `postselection_discard_rate: float | None`, `notes`, `error_message`. `PauliCheckOverhead`: `qubits_before`, `qubits_after`, `gates_before`, `gates_after`, `depth_before`, `depth_after` (ints).
  - `ibm_add_pauli_checks(inp) -> IbmAddPauliChecksOutput`.
- Consumes: nothing from other tasks.

- [ ] **Step 1: API discovery** in a scratch venv with `qiskit-paulice`: record how to add spacetime Pauli checks to a `QuantumCircuit` and how to read check outcomes for postselection. Put the notes in a comment block in `_pauli_checks_impl.py`.
- [ ] **Step 2: Failing tests** (`tests/test_ibm_pauli_checks.py`, `importorskip("qiskit_paulice")` for the real ones): `test_adds_checks_to_ghz3` (OpenQASM 2 GHZ-3: `qubits_after > qubits_before`, `gates_after > gates_before`, returned `circuit_qasm` parses), `test_stim_input_converted` (the same circuit as Stim text gives `ok=True`), `test_discard_rate_none_without_noise` and `test_discard_rate_in_unit_interval_with_noise` (`0 <= rate <= 1`), `test_invalid_qasm_reports_error` (`ok=False`, message mentions parsing), `test_oversized_circuit_rejected`, and in `test_base_install.py` the missing-extra case.
- [ ] **Step 3:** Run, expect FAIL. **Step 4: Implement** per signatures: parse with qiskit (QASM) or convert Stim via the existing qiskit conversion path in `ibm_hardware/_circuits.py` if one exists (reuse it, check first), call Paulice, compute overhead from `QuantumCircuit.num_qubits`, `count_ops`, `depth()`. Discard rate: sample the checked circuit with a depolarizing model only when `noise_p` is given and the circuit is Clifford (Stim); otherwise `None` plus a note.
- [ ] **Step 5:** Register in `server.py` (handler `_ibm_add_pauli_checks_handler`, `ToolSpec` description naming "spacetime Pauli checks", "postselection", "overhead"). Run tests, ruff, mypy. Expected: PASS.
- [ ] **Step 6:** chatQEC wiring; notebook section "7b. Pauli checks on IBM-style circuits" with GHZ-3 and the overhead table. Chat check: "Add spacetime Pauli checks to a 3-qubit GHZ circuit and show the overhead."
- [ ] **Step 7: Commit** both repos.

## Task 3: `ibm_ft_compile` (Qiskit FT transpile)

**Files:**
- Create: `tools/ibm_hardware/ft_compile.py`
- Modify: `server.py`, `pyproject.toml` (extra `ft = ["qiskit>=2.5"]`), `tests/test_base_install.py`
- Test: `tests/test_ibm_ft_compile.py`

**Interfaces:**
- Produces: `IbmFtCompileInput`: `circuit_qasm: str`, `mode: Literal["clifford_t", "pbc"] = "clifford_t"`, `gridsynth_precision: int = Field(10, ge=1, le=60)`. `IbmFtCompileOutput`: `ok`, `mode`, `t_count: int`, `clifford_count: int`, `depth: int`, `num_rotations: int | None` (PBC only), `circuit_qasm: str`, `notes`, `error_message`. `ibm_ft_compile(inp) -> IbmFtCompileOutput`.

- [ ] **Step 1: API discovery** with `qiskit>=2.5`: find the Clifford+T pass (gridsynth) and the PBC pass manager; confirm the version floor actually installs next to the existing `ibm` extra (`uv pip install --dry-run`). Notes as a comment block.
- [ ] **Step 2: Failing tests:** `test_rz_pi_over_4_is_one_t_gate` (single `rz(pi/4)` gives `t_count == 1` in `clifford_t`), `test_clifford_only_circuit_has_zero_t` (H, CX, S circuit), `test_pbc_reports_rotations`, `test_precision_changes_t_count` (higher precision never lowers `t_count` for `rz(0.3)`), `test_invalid_qasm_reports_error`, missing-extra case.
- [ ] **Step 3:** Run, expect FAIL. **Step 4:** Implement; counts from `count_ops()` (`t` + `tdg` as T). **Step 5:** Register `ToolSpec`; run tests, ruff, mypy.
- [ ] **Step 6:** chatQEC wiring; notebook section "7c. Fault-tolerant compilation". Chat check: "Compile this one-qubit rotation circuit to Clifford+T at precision 10 and give the T count."
- [ ] **Step 7: Commit** both repos.

## Task 4: `deltakit_memory` (Deltakit, local decoders only)

**Files:**
- Create: `tools/deltakit_memory.py`
- Modify: `server.py`, `pyproject.toml` (extra `deltakit = ["deltakit"]`), `tests/test_base_install.py`
- Test: `tests/test_deltakit_memory.py`

**Interfaces:**
- Produces: `DeltakitMemoryInput`: `code_family: str` (the families Deltakit code generation supports, fixed after discovery as a `Literal`), `distance: int = Field(3, ge=2, le=11)`, `rounds: int | None`, `p: float = Field(0.001, gt=0, le=0.1)`, `leakage_p: float = Field(0.0, ge=0, le=0.1)`, `decoder`: a `Literal` of the local decoders Deltakit exposes (including the leakage-aware local clustering one), `shots: int = Field(2000, ge=1, le=50_000)`, `seed: int | None`. `DeltakitMemoryOutput`: `ok`, `logical_error_rate`, `std_error`, `shots`, `num_qubits`, `num_detectors`, `decoder`, `notes`, `error_message`. Any cloud decoder name yields `ok=False` "cloud decoders are not configured".

- [ ] **Step 1: API discovery** with `deltakit` (0.8): code generation, noise (including leakage), the local decoders, how to run a memory experiment. Record which decoders are local and which need cloud credentials; only local ones go in the `Literal`.
- [ ] **Step 2: Failing tests:** `test_surface_memory_d3_runs` (`importorskip("deltakit")`; `ok`, `0 <= rate <= 1`, `num_detectors > 0`), `test_higher_p_not_lower_error` (p=0.001 vs 0.02, same seed, `rate_hi >= rate_lo`), `test_leakage_flag_accepted`, `test_cloud_decoder_rejected_with_message`, `test_even_distance_rules_match_family`, missing-extra case.
- [ ] **Step 3:** Run, expect FAIL. **Step 4:** Implement (lazy import, shot and distance caps, no network access attempted). **Step 5:** Register, run tests, ruff, mypy.
- [ ] **Step 6:** chatQEC wiring; notebook "7d. Deltakit memory experiments" including the "cloud decoders not configured" output as a line of text. Chat check: "Run a Deltakit memory experiment on a distance-3 surface code with leakage 0.001."
- [ ] **Step 7: Commit** both repos.

## Task 5: `guppy_run` (Guppy + Selene, sandboxed)

**Files:**
- Create: `tools/guppy_run.py`, `tools/_guppy_runner.py` (the script executed inside the sandbox)
- Modify: `server.py`, `pyproject.toml` (extra `guppy = ["guppylang", "selene-sim"]`), `tests/test_base_install.py`
- Test: `tests/test_guppy_run.py`

**Interfaces:**
- Consumes: `chatqec_mcp_tools.sandbox` (`SandboxConfig(wall_seconds, memory_mb, output_limit_kb)`, `run_in_sandbox`-style runner; read `sandbox.py` for the exact call name). Raise `memory_mb` for this tool only if Selene or ziglang needs more than the 1024 MB default, and record the chosen limit.
- Produces: `GuppyRunInput`: `source: str = Field(..., max_length=20_000)` (a Guppy program defining `main`), `shots: int = Field(100, ge=1, le=10_000)`, `seed: int | None`, `n_qubits: int = Field(4, ge=1, le=24)`. `GuppyRunOutput`: `ok`, `counts: dict[str, int]`, `shots`, `stderr_tail: str`, `error_message`. `guppy_run(inp) -> GuppyRunOutput`.

- [ ] **Step 1: API discovery** with `guppylang` and `selene-sim`: how to compile a Guppy `main` and run N shots on Selene's simulator, and how results are returned. Notes as a comment in `_guppy_runner.py`.
- [ ] **Step 2: Failing tests:** `test_bell_program_counts` (`importorskip("guppylang")`; a Bell-state Guppy program gives only `00` and `11` keys, counts sum to `shots`), `test_seed_reproducible` (same seed, same counts), `test_infinite_loop_times_out` (`ok=False`, message mentions the time limit), `test_hostile_source_cannot_read_home` (source tries `open(os.path.expanduser("~/.bashrc"))`: no file contents in the output, `ok=False`), `test_output_truncated`, `test_source_too_long_rejected`, missing-extra case.
- [ ] **Step 3:** Run, expect FAIL. **Step 4:** Implement: write `source` to a temp dir, run `_guppy_runner.py` through the sandbox with the limits above, parse a JSON result line from stdout. No network: run with an empty environment and, if the sandbox cannot block sockets, state that limitation in the tool description and the commit body.
- [ ] **Step 5:** Register, run tests, ruff, mypy.
- [ ] **Step 6:** chatQEC wiring; notebook "7e. Guppy programs on Selene". Chat check: "Write a Guppy program that prepares a Bell pair and run 200 shots on Selene."
- [ ] **Step 7: Commit** both repos.

## Task 6: BB5, tile and barbell parameters in `code_params`, plus RAG

**Files:**
- Create: `tools/_paper_codes.py` (static table)
- Modify: `tools/code_params.py` (`CodeParamsInput.code_family` Literal gains `"bb5"`, `"tile"`, `"directional_tile"`; `CodeParamsOutput` gains `source: str = ""` and `rows: list[dict] = []`)
- Test: `tests/test_code_params.py`, `tests/test_paper_codes.py`
- chatQEC: `sources/registry.jsonl` (new ingested papers), `src/chatqec/conversation/prompts.py`, notebook

**Interfaces:**
- Produces: `PaperCode(family: str, n: int, k: int, d: int, source: str, locator: str)` frozen dataclass; `paper_codes(family: str) -> list[PaperCode]`. `code_params` for these families returns every matching row (optionally filtered by `distance`) in `rows`, `source` set to the arXiv ids, no Stim distance computed.

- [ ] **Step 1: Ingest the missing sources first.** Directional tile codes: `uv run chatqec ingest --arxiv 2606.11484 --full-text`, check the registry title matches before embedding, then `uv run chatqec embed --source-id arxiv:2606.11484` (Gemini embedding provider, same as the collection). Barbell: search for the IQM preprint; if a verified preprint exists, ingest it the same way. If none exists, record that and add no barbell rows.
- [ ] **Step 2: Extract values.** Using `data/chunks/arxiv_2503.22071.jsonl` (BB5), `arxiv_2504.09171.jsonl` and `arxiv_2511.14589.jsonl` (tile), and the new ones, copy each stated `[[n,k,d]]` into `_paper_codes.py` with `source="arXiv:<id>"` and `locator` naming the table or section. Do not add a row you cannot point to.
- [ ] **Step 3: Failing tests:** `test_every_row_has_source_and_locator`, `test_every_source_is_in_registry` (reads `~/chatQEC/sources/registry.jsonl` via an env var path, skipped when absent), `test_k_and_d_positive`, `test_code_params_bb5_returns_rows`, `test_code_params_distance_filter`, `test_barbell_not_offered_without_source` (no `"barbell"` in the `Literal` unless rows exist).
- [ ] **Step 4:** Run, expect FAIL. **Step 5:** Implement; run tests, ruff, mypy.
- [ ] **Step 6:** chatQEC: prompt bullet saying to prefer `code_params` for BB5 and tile parameters and to cite the `source`; retrieval check `uv run chatqec retrieve --query "BB5 code parameters"` returns the BB5 paper; notebook "7f. Codes from the literature". Chat check: "What are the parameters of BB5 codes?" The answer should cite `arXiv:2503.22071`.
- [ ] **Step 7: Commit** both repos (registry and chunk files in the chatQEC commit only if they are not gitignored; check `git check-ignore data/chunks`).

## Task 7: End-to-end acceptance, docs, notebook

**Files:** `README.md` (tool list and install extras), `CHANGELOG.md`, `notebooks/getting_started.ipynb`, `docs/EXAMPLES.md`.

- [ ] **Step 1:** Install everything: `uv tool install --force ~/chatqec-mcp-tools[lightstim,tsim,ibm,decoders,cudaq,paulice,ft,deltakit,guppy]`; restart Streamlit with `CHATQEC_PROVIDER=local CHATQEC_MCP_SERVER=chatqec-mcp-tools uv run streamlit run chatqec_app/streamlit_app.py`.
- [ ] **Step 2:** In the Streamlit chat (use the `webapp-testing` skill/Playwright against :8501), run the two acceptance prompts on Muse-Glimmer, then Gemma 4:
  - Flow 1: "Compare PyMatching, Tesseract, MLE and CUDA-Q RelayBP on a distance-5 color code at p=0.002 and rank them."
  - Flow 2: "Add spacetime Pauli checks to this OpenQASM GHZ-3 circuit and report qubit, gate and depth overhead."
  Expected: each shows a tool call, a rendered result, and a correct summary. Save screenshots under `docs/` only if the repo already keeps them; otherwise attach nothing. A model that never calls the tool is a prompt or router defect to fix.
- [ ] **Step 3:** Update `README.md` (new tools and extras, replacing "QEC simulators" row text where needed), `CHANGELOG.md`, `docs/EXAMPLES.md` with one example prompt per new tool.
- [ ] **Step 4:** Execute the whole notebook top to bottom (`jupyter nbconvert --execute --inplace`) and confirm no cell errors; the "More prompts to try" table lists one prompt per new tool.
- [ ] **Step 5:** `uv run pytest -q` and `uv run ruff check .` in both repos. Expected: green apart from the pre-existing unrelated eval-work failures noted in Task 0.
- [ ] **Step 6: Commit** (`docs: document the integrated vendor tools and refresh the getting-started notebook`).

## Out of scope here

The benchmark work (decoder-bench, StabilizerBench, FTCircuitBench, FTPrimitiveBench ingestion and evaluation) follows after Task 7 under the approved benchmark plan at `~/.claude/plans/add-the-following-on-hashed-globe.md`; its design-doc section is appended then.
