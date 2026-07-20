# QEC Tooling Integration — PyMatching, qiskit-qec, Threshold Sweep

**Date:** 2026-07-12
**Status:** Approved (design)
**Scope:** Add three QEC tools to the `chatqec-mcp-tools` MCP sibling, plus main-repo
prompt + Streamlit wiring so they are usable end to end.

## Motivation

The MCP sibling (`../chatqec-mcp-tools`) already ships `stim_simulate` and
`stim_diagram`. Its own roadmap names the next three tools: `pymatching_decode`,
`qiskit_qec_run`, `threshold_sweep`. These let the assistant *decode* circuits,
*inspect* code parameters, and *find thresholds* with real computation instead of
guessing numbers — the same "compute, don't hallucinate" motivation behind the
Stim tools.

## Architecture

Mirror the existing Stim pattern exactly:

- Each tool is one file `src/chatqec_mcp_tools/tools/<name>.py` exposing a pydantic
  `Input`/`Output` model (both `extra="forbid"`) and a pure
  `fn(inp: Input) -> Output` that: guards input size, lazy-imports the heavy lib,
  and returns `ok=True/…` or `ok=False, error_message=…`. No exception escapes to
  the MCP layer.
- Registered as a `ToolSpec` in `server.py:_TOOLS` (name, description,
  `Input.model_json_schema()`, handler). Auto-discovered by the main app over
  stdio via `list_tools()` — no allowlist to update.
- The existing `sandbox.py` (subprocess + rlimits) is **not** used; all three tools
  run in-process behind input guards, exactly like the Stim tools.

### Dependency policy

- `pymatching>=2.0` → **core** dependency (light C++ extension; shared by two tools).
- `qiskit-qec` → **optional extra** `[qiskit]` (drags in full qiskit; less actively
  maintained). The tool lazy-imports and returns a friendly
  `install chatqec-mcp-tools[qiskit]` error when absent — same graceful-degrade
  pattern as the main repo's `vision` extra.

## Components

### Shared decode helper — `tools/_decoding.py`

`pymatching_decode` and `threshold_sweep` share one internal primitive so the
Stim+PyMatching decode logic lives in exactly one place:

```
_decode_circuit(circuit: stim.Circuit, shots: int) -> DecodeStats
    # DecodeStats: logical_error_rate, num_errors, shots, num_detectors, num_observables
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    det, obs = circuit.compile_detector_sampler().sample(shots, separate_observables=True)
    pred = matching.decode_batch(det)
    num_errors = (pred != obs).any(axis=1).sum()   # any observable mismatch = logical error
    ...
```

### Tool 1 — `pymatching_decode`

- **Input:** `circuit: str` (Stim source, must contain detectors *and* observables),
  `shots: int` (`ge=1`, `≤ 1_000_000`).
- **Flow:** parse circuit → `_decode_circuit`.
- **Output:** `ok`, `logical_error_rate: float`, `num_errors: int`, `shots: int`,
  `num_detectors: int`, `num_observables: int`, `error_message: str = ""`.
- **Guards:** 256 KB circuit cap and 1e6 shot cap (reuse the Stim constants);
  `error_message` on stim `ValueError`, zero observables, or DEM-decomposition failure.

### Tool 2 — `threshold_sweep`

- **Input:** `code_family: Literal["surface_code_rotated", "repetition_code", "color_code"]`,
  `distances: list[int]` (1–5 entries, each in `3..15`, odd enforced where the
  family requires it),
  `p_values: list[float]` (1–12 entries in `(0, 0.5]`),
  `rounds: int | None` (default = distance), `shots: int` (`≤ 1e6`).
- **Flow:** for each `(d, p)` build `stim.Circuit.generated(<mapped family>, distance=d,
  rounds=rounds or d, after_clifford_depolarization=p,
  before_measure_flip_probability=p, after_reset_flip_probability=p)`; decode via the
  shared primitive.
- **Output:** `ok`, `rows: list[{distance, p, shots, logical_error_rate}]`,
  `plot_svg: str` (matplotlib log-log plot, one line per distance — the threshold is
  the crossing point), `error_message`.
- **Guards:** reject if `shots · len(distances) · len(p_values) > 20_000_000` (the
  work ceiling); validate family/distance/p ranges before any sampling. `plot_svg`
  capped at the 100 KB output limit.

Family name mapping: `surface_code_rotated → "surface_code:rotated_memory_z"`,
`repetition_code → "repetition_code:memory"`, `color_code → "color_code:memory_xyz"`.

### Tool 3 — `qiskit_qec_run`

- **Input:** `code_family: str` (e.g. `surface`, `repetition`), `distance: int`.
- **Flow:** construct the code via qiskit-qec; read off its parameters. No arbitrary
  code execution.
- **Output:** `ok`, `n: int`, `k: int`, `d: int`, `stabilizers: list[str]` (Pauli
  strings), `logical_x: list[str]`, `logical_z: list[str]`, `error_message`.
- **Risk / open item:** the exact qiskit-qec constructor API is *pinned during
  implementation* (TDD, against the installed version). A family that cannot be
  cleanly constructed is **omitted from the accepted set rather than faked**. If the
  extra is not installed the tool returns the install-hint error.

## Data flow

`main app (ask --mcp-server / Streamlit)` → stdio → `chatqec-mcp-tools server` →
`ToolSpec.handler` → `Input.model_validate` → pure `fn` → `Output.model_dump` →
JSON `TextContent` → main app renders.

## Main-repo integration (full scope)

- **Prompt** (`src/chatqec/conversation/prompts.py`): add one paragraph naming the
  three tools and when to reach for each — decode a circuit → `pymatching_decode`;
  find/plot a threshold → `threshold_sweep`; inspect a code's `[[n,k,d]]`/stabilizers
  → `qiskit_qec_run`.
- **Streamlit** (`chatqec_app/components/tool_output.py`): render `threshold_sweep`'s
  `plot_svg` inline (extend the existing `<svg>` branch) and its `rows` as a table;
  render `pymatching_decode` / `qiskit_qec_run` as compact metric/JSON blocks.

## Error handling

Uniform `ok=False, error_message=…` for: oversized input, invalid/unparseable
circuit, missing detectors/observables, DEM-decomposition failure, unknown code
family, out-of-range distance/p, work-ceiling exceeded, and missing `[qiskit]` extra.
The MCP `call_tool` layer never sees a raised exception.

## Testing (TDD, per tool)

- `pymatching_decode`: real fast case — repetition `d=3`, ~200 shots — assert
  `0 ≤ ler ≤ 0.5` and correct detector/observable counts; guard-path tests
  (oversized circuit, circuit with no observables).
- `threshold_sweep`: small sweep (repetition, `distances=[3,5]`, 2 p-values, ~200
  shots); assert row count = `|distances|·|p_values|`, each `ler` in range, `plot_svg`
  contains `<svg`; work-ceiling rejection test.
- `qiskit_qec_run`: `pytest.importorskip("qiskit_qec")` for the real construction path
  (assert `n,k,d` for a known small code) **and** an explicit test of the missing-extra
  friendly error.
- `server.py`: assert all three names appear in `build_server().list_tools()` and each
  handler round-trips a minimal valid input.

## Out of scope (YAGNI)

Other roadmap tools (`stabilizer_check`, `code_search`, `logical_error_curve`,
`decode_compare`, `monte_carlo_estimate`); Docker sandbox; check-matrix/syndrome input
form for `pymatching_decode` (Stim-circuit input only).

## Decisions locked

1. `pymatching_decode` input is a **Stim circuit**, not a raw check-matrix + syndrome.
2. `qiskit-qec` is an **optional extra**, not a core dependency.
3. `threshold_sweep` returns **table + SVG plot**.
4. `qiskit_qec_run` does **code construction + parameters** only.
