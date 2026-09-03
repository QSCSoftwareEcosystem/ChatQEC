# IBM Hardware-Aware Circuit Mapping and Calibration Q&A — Design

**Date:** 2026-09-03
**Status:** Draft — pending user review
**Scope:** Give ChatQEC three new capabilities backed by real IBM Quantum data: (1) answer
questions about a specific IBM backend's calibration/noise, (2) describe a specific IBM
architecture (topology, qubit count, coupling map), and (3) given a circuit (Bell, GHZ,
QAOA, or arbitrary QASM), map it onto a chosen or auto-selected IBM backend and, optionally,
run it for real. This closes the "Hardware-aware placement... no implementation exists yet"
gap in `README.md:496-501,664-671`.

**Repos touched:** `chatqec-mcp-tools` only. No changes needed in `chatQEC` itself — new
tools appear automatically once the installed `chatqec-mcp-tools` package is upgraded,
because `conf/config.yaml:57` already points `server_command` at it and `tool_loop.py`
surfaces whatever tools the server advertises.

## Motivation

ChatQEC already has a working pattern for adding capability via MCP tools (`stim_simulate`,
`code_params`, etc., all in `chatqec-mcp-tools`) and a model-driven tool loop that decides
when to call them. It has zero quantum-hardware awareness today: no Qiskit dependency, no
notion of a physical backend, no calibration data. A prior intern project
(`/home/pza/intern/sangram/HardwareAwarenessIntegrationPipeline`) already explored most of
the hard parts — live IBM calibration fetch, a normalized hardware graph, and three
placement strategies (VF2 exact match, Steiner tree, weighted greedy) with genuine
objective-based quality scoring — but is tangled with unrelated LLM/QEC-encoder/benchmark
code, has no packaging boundary, no tests, and one specific weakness worth fixing rather
than copying (see "Mapping algorithm" below).

## Why not the obvious alternatives

**Why not import the intern's repo as a dependency as-is.** It pins `sys_platform=='darwin'`
and Python `3.13,<3.14` (won't install on this Linux/HPC stack), has no test suite, mixes
29 MB of generated `output/` and 31 MB notebooks into the source tree, and its most
LLM-driven layout arm (`llm_optimizer.py`, `layout_search.py`) was flagged by its own prior
audit (`internal_memos/auditing/phase_4,7`) as the least mature part. Porting the sound
pieces (hardware graph schema, quality model, greedy+VF2 placement) and dropping the rest is
less risk than depending on the whole thing.

**Why not just call Qiskit's built-in transpiler with defaults and stop there.** That would
answer "can this circuit run on this backend" but not "which physical qubits should Claude
recommend and why" — the objective-scored placement (favor low-error couplers, avoid
short-T1 qubits) is the actual value add over Sabre's generic layout, and it's what the user
explicitly asked for ("careful about the intern's mapping technique; invent novel and
efficient technique if necessary").

**Why not put the new code inside `src/chatqec/` directly (in-process, no MCP).** Breaks
from the established pattern where all circuit/simulator tools live in `chatqec-mcp-tools`
and are exposed over MCP stdio; would also force a `qiskit` dependency onto the core
retrieval package, which is expressly avoided today (`code_params.py` docstring: "No
qiskit-qec dependency").

**Why not always fetch calibration live.** IBM Runtime backend-properties calls are free
(no queue, no shots) but still a network round trip against a rate-limited free-tier
account, and calibration does not change every few minutes. A local TTL cache (new to this
repo — no existing cache pattern to reuse) avoids hammering the API on every chat turn while
staying honest by always reporting the data's age.

## Architecture

```
user question
   |
   +--> Claude (tool_loop.py) decides which tool(s) to call
          |
          +--> ibm_list_backends            (free, short-cached)
          +--> ibm_backend_calibration       (free, TTL-cached, reports fetched_at/age)
          +--> ibm_map_circuit               (free, local compute + coupling-map SVG)
          +--> ibm_preview_execution         (free, no submission)
          +--> ibm_submit_execution          (SPENDS CREDITS, requires prior preview)
                |
                v
        qiskit_ibm_runtime.QiskitRuntimeService
        (QISKIT_IBM_TOKEN / QISKIT_IBM_INSTANCE env vars only)
```

All five tools live in `src/chatqec_mcp_tools/tools/ibm_hardware/`, registered in
`server.py`'s `_TOOLS` list exactly like every existing tool (thin `_handler` wrapping
`Input.model_validate(args)` → function → `.model_dump()`).

## Module layout

```
tools/ibm_hardware/
  __init__.py
  list_backends.py        # ibm_list_backends
  calibration.py          # ibm_backend_calibration
  map_circuit.py          # ibm_map_circuit
  preview_execution.py    # ibm_preview_execution
  submit_execution.py     # ibm_submit_execution
  _client.py              # lazy QiskitRuntimeService construction from env vars
  _cache.py               # new: file-based TTL cache (JSON, fetched_at/ttl_seconds/payload)
  _hardware_graph.py       # ported from cahgr/hardware_kg.py: calibration -> normalized graph
  _quality_model.py        # ported from cahgr/retrieval.py: HardwareQualityModel
  _placement.py            # ported+fixed: greedy region-growing + bounded VF2
  _coupling_diagram.py     # matplotlib -> SVG string, mapped qubits highlighted
```

`pyproject.toml` gains an optional extra:

```toml
[project.optional-dependencies]
ibm = ["qiskit>=1.0", "qiskit-ibm-runtime>=0.48"]
```

`qiskit`/`qiskit_ibm_runtime` are imported lazily inside function bodies (same pattern as
`stim`/`pymatching` elsewhere in this repo), so the base package still installs and every
non-`ibm_*` tool still works without the extra.

## Tool specs

### `ibm_list_backends`

```
Input:  {}                                    # no args; uses configured account
Output: {ok, backends: [{name, num_qubits, operational, pending_jobs, basis_gates}],
         error_message}
```
Cached 60s (backend status changes fast; this is cheap to refetch, unlike calibration).

### `ibm_backend_calibration`

```
Input:  {backend: str}
Output: {ok, backend, fetched_at: str (ISO8601), age_seconds: int,
         qubits: [{index, t1_us, t2_us, readout_error}],
         couplers: [{qubits: [int, int], two_qubit_error, gate_duration_ns, native_gate}],
         error_message}
```
TTL-cached (default 3600s, override via `CHATQEC_IBM_CALIBRATION_TTL_SECONDS`). Every answer
built from this must surface `age_seconds` in prose ("fetched 12 min ago" / "fetched just
now") — this is a synthesizer-side convention, not something the tool enforces, since the
tool only returns data.

### `ibm_map_circuit`

```
Input:  {circuit_qasm: str | None, circuit_preset: Literal["bell","ghz","qft","qaoa"] | None,
         num_qubits: int | None,       # for presets
         qaoa_layers: int | None,      # for the qaoa preset
         backend: str | None,          # None = auto-select
         objective: Literal["maximum_fidelity","minimum_execution_time"] = "maximum_fidelity"}
         # exactly one of circuit_qasm / circuit_preset required
Output: {ok, backend, physical_qubits: list[int], logical_to_physical: dict[int, int],
         strategy_used: Literal["vf2","greedy"], transpiled_depth: int,
         two_qubit_gate_count: int, estimated_fidelity: float,
         coupling_map_svg: str, fetched_at, age_seconds, error_message}
```

### `ibm_preview_execution`

```
Input:  {circuit_qasm | circuit_preset (as above), backend: str, shots: int = 1000,
         objective: ... = "maximum_fidelity"}
Output: {ok, backend, mapping: <same shape as ibm_map_circuit output minus svg>,
         shots, estimated_queue_position: int | None, backend_operational: bool,
         error_message}
```

### `ibm_submit_execution`

```
Input:  {circuit_qasm | circuit_preset (as above), backend: str, shots: int}
Output: {ok, job_id: str, backend, shots, status: str, error_message}
```
Caps enforced before submission (all violations return `ok=False` with `error_message`,
never raise): `num_qubits <= 10`, `shots <= 4000`, `backend` required (no auto-pick). No
`confirm` flag — the tool's existence is itself the confirmation step; a model can only
reach it after already having called `ibm_preview_execution`, which is documented in the
tool's MCP description ("call ibm_preview_execution first to see the plan before spending
credits") and enforced socially via that description, not technically, since MCP tools are
stateless between calls and this repo has no session/state layer to enforce call ordering
server-side. This is a documented limitation (see below), not a hard guarantee.

## Mapping algorithm

Ported from `cahgr/hardware_kg.py` and `cahgr/retrieval.py`, generalized from the intern's
fixed prompt-library circuits to arbitrary user-supplied QASM:

- **`_hardware_graph.py`**: calibration JSON → a graph with per-qubit (T1, T2, readout
  error) and per-coupler (two-qubit error, gate duration) attributes. Kept as-is; the review
  found this schema sound.
- **`_quality_model.py`**: min-max normalized objective scoring (`maximum_fidelity` weights
  T1/T2/readout/gate-fidelity; `minimum_execution_time` switches to gate duration). Kept
  as-is.
- **`_placement.py`**: two strategies.
  - **Greedy region-growing** (default, always succeeds): grow a candidate physical-qubit
    region from a high-quality seed qubit, scored by the objective model.
  - **VF2 exact subgraph match**: attempted first when the circuit's two-qubit interaction
    graph is small enough. This is the one piece that gets an actual fix rather than a
    straight port: the intern's version capped search at a hardcoded 12 interaction-graph
    nodes / 4000 candidates with no stated justification. Replacing that: candidates are
    generated in order of a cheap greedy pre-score (best regions first) and the search stops
    on a **time/candidate budget** (default: 2 seconds or 2000 candidates, whichever first),
    falling back to greedy automatically when the budget is exhausted or the interaction
    graph is too large to match exactly. This makes the exact-match effort scale with actual
    difficulty instead of an arbitrary constant, and it always terminates with *some* valid
    mapping (greedy) even when VF2 finds nothing.
  - **Steiner-tree placement is dropped**, not ported. Greedy + bounded VF2 cover the
    exact/approximate spectrum for Bell/GHZ/QAOA-scale circuits (the actual target use
    case); Steiner adds real complexity for a case not yet observed to need it. If a future
    circuit shape needs it, `_placement.py` is structured so a third strategy slots in next
    to the other two without touching the tool layer.

Dropped, not ported: the LLM circuit-generation layer (Claude's tool loop already writes
QASM from a natural-language description directly), the `[[5,1,3]]` QEC encoder, the
benchmarking harness (817 lines, scientifically unvalidated per the prior audit), and the
LLM-aware layout-search arm.

## Caching

New pattern for this repo (no existing cache to model). File-based, JSON, one file per
`(kind, backend)` key under `$XDG_CACHE_HOME/chatqec-mcp-tools/ibm_hardware/` (falls back to
`~/.cache/...`), each holding `{fetched_at, ttl_seconds, payload}`. `_cache.py` exposes
`get_or_fetch(key, ttl_seconds, fetch_fn) -> (payload, fetched_at, age_seconds)`, checked
before any IBM API call. Default TTLs: 3600s for calibration, 60s for backend list/status.
Both overridable via env vars. Cache persists across chat sessions (each session spawns a
fresh MCP server subprocess, so an in-memory cache would not survive between sessions;
disk-based avoids repaying the fetch cost every new conversation).

## Verified against your live IBM account (2026-09-03)

Checked live with `qiskit-ibm-runtime` 0.49.0 against the free "open" plan account, so the
plan below can rely on these rather than assumptions:

- **Only `ibm_quantum_platform`/`ibm_cloud` channels exist** in this SDK version; the legacy
  `ibm_quantum` channel referenced by the intern's code and by older docs has been fully
  retired and raises `ValueError` if passed.
- **No `QISKIT_IBM_INSTANCE` is required.** `QiskitRuntimeService(channel="ibm_quantum_platform", token=...)`
  auto-selects the account's single `open-instance` on the free plan (with a warning, not an
  error) — `_client.py` can rely on this default and only needs `QISKIT_IBM_INSTANCE` for
  accounts with more than one instance.
- **3 backends currently visible**, all 156-qubit Heron processors: `ibm_fez` (0 pending
  jobs), `ibm_marrakesh` (3 pending), `ibm_kingston` (3 pending). `ibm_list_backends` must
  reflect whatever the account can see dynamically — confirmed there is no fixed set to
  hardcode, and the free-tier lineup can differ from the intern's pinned snapshot set
  (`ibm_boston`, absent from `ibm_data/`'s stale snapshot, isn't visible here at all;
  `ibm_marrakesh` is the one backend present in both the live account and the pinned
  fixtures, useful for integration-test continuity).
- **A coupling pair can expose more than one native two-qubit gate.** `ibm_fez` qubits (0, 1)
  list both `cz` and `rzz` as available two-qubit basis gates — `_hardware_graph.py` must
  store a per-coupler *list* of (gate name, error, duration) tuples, not a single native
  gate, correcting an implicit single-gate assumption in the intern's schema.
- **Calibration data was ~33 minutes old** at check time (`last_update_date` timestamp),
  consistent with same-day (sub-daily) recalibration — supports the 3600s default TTL as
  reasonable rather than needing to guess.

## Credentials

`QISKIT_IBM_TOKEN` and `QISKIT_IBM_INSTANCE` read from environment only, lazily, inside
`_client.py`, matching `hardware_pull/vendor_ibm.py`'s existing pattern in the intern repo
(never a tool input parameter, never written to a tracked file, per `rules/security.md`).
If unset, every IBM tool returns `ok=False` with a clear `error_message` rather than
raising. Falls back to a saved `qiskit-ibm-runtime` account (`QiskitRuntimeService()` with
no args) if no env token is present, so a one-time `service.save_account(...)` also works.

## Error handling

| Failure | Behavior |
|---|---|
| `QISKIT_IBM_TOKEN` not set and no saved account | `ok=False`, `error_message` explains how to set it; never raises |
| IBM API unreachable / times out | `ok=False` with the underlying error message; cached data (if any, even if stale) is preferred with a note that it's stale, rather than failing outright, for `ibm_backend_calibration` |
| Requested backend does not exist / not visible to this account | `ok=False`, lists available backend names from `ibm_list_backends`-equivalent lookup |
| Circuit uses more qubits than the backend has | `ok=False` from `ibm_map_circuit`, no fallback |
| VF2 budget exhausted, no exact match found | Falls back to greedy silently; `strategy_used: "greedy"` in output tells the caller which happened |
| `circuit_qasm` fails to parse | `ok=False` with the parser error, no partial result |
| Both or neither of `circuit_qasm`/`circuit_preset` given | Pydantic validation error (rejected before the function body runs, `extra="forbid"` style) |
| `ibm_submit_execution` caps violated (qubits/shots/no backend) | `ok=False`, explains which cap failed, nothing submitted |
| Cache file corrupt | Treated as a cache miss, refetched, file rewritten |

## Testing

Mirrors the existing per-tool convention: `tests/test_<tool>.py` importing
Input/Output/function directly, IBM API calls mocked (no live network in unit tests).
- `_placement.py`: unit tests against small synthetic coupling graphs (line, ring, heavy-hex
  fragment) checking greedy always returns a valid mapping, VF2 finds an exact match when
  one exists and correctly falls back when the budget is set to zero.
- `_quality_model.py`: known-input/known-output cases for both objectives.
- `_cache.py`: TTL expiry, corrupt-file recovery, key isolation between backends.
- Calibration/backend fixtures: the six pinned JSON snapshots from the intern's
  `hardware_data/` directory, repurposed as offline test fixtures (explicitly never treated
  as live data, since the prior audit found them ~50 days stale).
- `test_server.py`: registration assertions for all five new tools, following the existing
  `build_server()` round-trip pattern.
- One live, manual smoke test (not part of CI): with `QISKIT_IBM_TOKEN` set, run
  `ibm_list_backends` and `ibm_backend_calibration` against the real account once, to confirm
  the account's actual accessible backend(s) and shape the fixtures/defaults realistically.
  This has not been run yet — no token is available in the current environment.

## Known limitations

- **`ibm_submit_execution`'s "preview first" rule is enforced only by tool description, not
  by server-side state.** MCP tools here are stateless; there is no session concept to
  verify a preview actually happened first. A determined or confused model could call submit
  directly. The qubit/shot caps and required explicit backend are the real safety net; the
  preview-first tool split is a strong nudge, not a guarantee.
- **Calibration TTL of 3600s is a starting guess** informed by one observation (~33 min old
  at check time), not a measured recalibration cadence; easy to tune via env var once more
  data is observed over time.
- **No near-duplicate or job-history tracking.** Repeated `ibm_submit_execution` calls are
  not deduplicated; a user or model calling it twice submits two jobs.

## Benchmark prompts (done ahead of implementation)

Added 2026-09-03: 12 hardware-aware Q&A cases (`hw-001`..`hw-012`, new `hardware_mapping`
subtype) to `eval/qa_set.jsonl`, covering calibration-fact lookup, IBM architecture/topology
description, and mapping-technique reasoning (Bell/GHZ/QAOA), for future evaluation of
ChatQEC (or a comparable chatbot) once this feature exists. No evaluation run yet — this is
benchmark data only. Numeric calibration facts are grounded in the intern's pinned
`hardware_data/ibm_boston.json` / `ibm_marrakesh.json` snapshots (dated 2026-07-15) rather
than live data, so the gold answers stay stable regardless of when the eval is actually run;
each such case's question is phrased to reference "the pinned 2026-07-15 snapshot"
explicitly. `QuestionSubtype` gained `HARDWARE_MAPPING = "hardware_mapping"`; the enum-size
test in `tests/test_eval_dataset.py` and the eval-set-size mentions in `README.md`
("50-pair" -> "62-pair", "6 subtypes" -> "7 subtypes") were updated to match. Future prompts
for this feature area should keep using the `hardware_mapping` subtype and land in the same
file as the implementation progresses.

## Out of scope

Steiner-tree placement (documented above as a future add-on if needed); a UI-visible job
queue/history view in `chatqec_app/`; multi-backend comparison in a single tool call (call
`ibm_backend_calibration` per backend instead); IQM or any non-IBM hardware target, despite
being mentioned as a future item in `README.md:499` (this spec is IBM-only); automatic
retry/resubmission of failed real-hardware jobs.
