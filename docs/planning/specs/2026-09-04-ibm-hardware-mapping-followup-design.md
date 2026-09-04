# IBM Hardware Mapping — Follow-Up: Real Numbers, Reconciled Caps, Working Diagram — Design

**Date:** 2026-09-04
**Status:** Draft — pending user review
**Scope:** Close four gaps found by the final whole-branch review of the original
`2026-09-03-ibm-hardware-mapping` feature, deliberately parked out of that review's one fix
wave because they needed real design decisions (not one-line bug fixes) and, in one case,
cross-repo work. This spec covers: making `ibm_map_circuit`'s reported numbers reflect real
transpilation, making the default hardware-quality objective actually consider T1/T2,
reconciling `ibm_preview_execution` and `ibm_submit_execution` so a preview can't show a plan
submit then refuses, and fixing the coupling-map diagram so it actually renders in chatQEC's
UI instead of silently costing ~26k tokens per call with nothing to show for it.

**Repos touched:** `chatqec-mcp-tools` (all four fixes) and `chatQEC` (the diagram-rendering
half of the fourth fix only).

## Motivation

The original feature's final review (`docs/planning/plans/2026-09-03-ibm-hardware-mapping.md`,
final-review section of that plan's SDD ledger) found these as Important-severity findings,
distinct from the two Critical bugs fixed in that review's one allowed fix wave:

- **I4:** `ibm_map_circuit`'s `transpiled_two_qubit_gate_count` field is computed from the
  *un-routed* logical circuit — no transpilation happens anywhere in that tool. A placement
  needing SWAPs reports a materially wrong (too-low) gate count, and the field name itself
  claims something the tool doesn't do. The design's own declared output field
  `transpiled_depth` was never implemented at all.
- **I6:** the default `"maximum_fidelity"` objective narrows node scoring to readout fidelity
  alone, so T1/T2 don't affect placement under the objective every tool defaults to — directly
  contradicting the original spec's own pitch ("favor low-error couplers, avoid short-T1
  qubits") and two committed benchmark gold answers (`hw-006`, `hw-010` in
  `eval/qa_set.jsonl`), which both describe a mapper that weighs T1/T2 alongside readout.
- **I5:** `ibm_preview_execution` accepts `num_qubits` up to 1024 and unbounded `shots`, and
  lets the caller pick an `objective`; `ibm_submit_execution` caps at 10 qubits / 4000 shots
  and hardcodes `objective="maximum_fidelity"`. A preview can show a plan that submit then
  refuses, or a plan built under a different objective than what actually gets submitted —
  undermining the entire "preview before you spend credits" safety story.
- **I3:** `render_coupling_diagram` renders the *whole* hardware graph. On any of this
  account's real backends (all three visible ones are 156-qubit devices, verified live during
  the original spec's design phase), that produces a ~96KB SVG (~26,000 tokens once
  JSON-serialized into the tool result). Worse, `chatQEC`'s `tool_output.py` has no rendering
  branch for `ibm_map_circuit` at all — every other SVG-bearing tool (`stim_diagram`,
  `threshold_sweep`) is rendered to the user; this one silently isn't, contradicting the
  system prompt's explicit promise that diagrams are "rendered to the user separately, ALWAYS
  below your answer text." The tokens are spent for a diagram nobody ever sees.

**Why fix these now rather than leave them parked indefinitely.** I4 and I6 both mean the
tool can state a materially wrong number to a user with full confidence — exactly the
failure mode this whole feature exists to prevent. I5 undermines the credit-safety design.
I3 means the single most visual deliverable of the feature (a device diagram with the mapped
qubits highlighted) has never worked, on every real backend this account can see.

## Design

### I6 — T1/T2-aware `maximum_fidelity` objective (`_quality_model.py`)

`HardwareQualityModel.node_score`'s `if self.objective == "maximum_fidelity":` branch
currently returns `self._ro_norm(readout_fidelity)` alone. **Remove that special case.** Use
the same formula already used for every other objective — the mean of whichever of
T1-normalized, T2-normalized, and readout-fidelity-normalized are present (falling back to
`0.5` if none are) — unconditionally, regardless of `objective`. `edge_score` is unaffected:
`minimum_execution_time` still switches to gate-duration scoring; every other objective
(including `maximum_fidelity`) still uses gate-fidelity.

This is a strict simplification (one formula instead of two subtly different ones) as well
as a correctness fix, and it requires no change to the two benchmark prompts — their gold
answers already describe the behavior this produces.

### I4 — real transpilation in `ibm_map_circuit`

**New shared module** `_transpile.py` in `tools/ibm_hardware/`, extracting the existing
`_transpile(circuit, backend, initial_layout) -> Any` helper out of `submit_execution.py`
(a thin lazy wrapper around `qiskit.transpile`, unchanged in behavior) so both
`map_circuit.py` and `submit_execution.py` import it from one place instead of
`submit_execution.py` owning the only copy.

In `compute_mapping` (`map_circuit.py`), after `place()` returns a placement: call
`_transpile(spec.circuit, sdk_backend, initial_layout)` — no `measure_all()` call here (unlike
`ibm_submit_execution`), since this tool never runs anything; measurements are irrelevant to
depth/gate-count/fidelity analysis and adding them would change the reported metrics for no
reason. From the resulting `isa_circuit`:

- `transpiled_depth: int = isa_circuit.depth()` — a real spec-declared field that was never
  implemented; now it is.
- `transpiled_two_qubit_gate_count`: count real 2-qubit operations in `isa_circuit.data`
  (excluding barrier/measure/reset/delay, matching `_circuits.py`'s existing exclusion set —
  this is also where the M2 minor from the final review, the gate-exclusion set being
  hand-duplicated between `_circuits.py` and `map_circuit.py`, gets fixed as a side effect:
  export the frozenset from `_circuits.py`, import it in both places instead of duplicating).
- `estimated_fidelity`: recomputed from the *real* transpiled circuit's actual 2-qubit gates
  and the physical qubits they land on, looking up each gate's real `gate_fidelity` from
  `hw_graph`, plus each used physical qubit's `readout_fidelity` (note for whoever implements
  this: after `transpile(circuit, backend=sdk_backend, initial_layout=...)`, `isa_circuit`'s
  qubit-register indices directly correspond to the backend's physical qubit numbers,
  confirmed empirically during the original feature's design phase via
  `isa.layout.initial_index_layout()` — so each instruction's qubit indices can be used
  directly as `hw_graph` node/edge keys, no extra layout-translation step needed) — same
  product-of-independent-fidelities approach as today, just computed from what will
  actually run instead of from the pre-routing interaction graph. This automatically fixes
  the final review's finding that the old computation silently skipped SWAP-requiring edges
  (a SWAP-heavy placement now correctly shows a lower fidelity, because the real SWAP gates
  are now in the circuit being scored).

No change to `ibm_preview_execution`/`ibm_submit_execution`'s own behavior beyond importing
`_transpile` from its new shared location instead of defining it locally.

### I5 — preview/submit reconciliation

**`ibm_preview_execution`** gains `within_submit_caps: bool` on its output: `True` iff
`len(physical_qubits) <= SUBMIT_MAX_QUBITS and shots <= SUBMIT_MAX_SHOTS` (both from the new
shared `_limits.py`, see below). This lets the model tell a user "this plan previews fine but
exceeds what `ibm_submit_execution` allows" instead of the user finding out only after
attempting to submit.

**`ibm_submit_execution`** gains an `objective: str = "maximum_fidelity"` input field
(validated against the same two-value set `map_circuit.py` already validates against),
threaded into its `compute_mapping(...)` call instead of the current hardcoded literal — so
a caller can preview and submit under the same objective and get the same placement (modulo
the VF2 time-budget's inherent, already-documented run-to-run variance on match-rich inputs).

**New shared module** `_limits.py` in `tools/ibm_hardware/`, consolidating the four `_MAX_*`
constants currently duplicated (with matching values, verified consistent, but confusingly
similar names like `_MAX_NUM_QUBITS` vs `_MAX_QUBITS`) across `map_circuit.py`,
`preview_execution.py`, and `submit_execution.py`:

```python
MAX_PRESET_QUBITS = 1024       # bound on num_qubits for bell/ghz/qft/qaoa presets
MAX_QAOA_LAYERS = 1000         # bound on qaoa_layers
SUBMIT_MAX_QUBITS = 10         # ibm_submit_execution's physical-qubit cap
SUBMIT_MAX_SHOTS = 4000        # ibm_submit_execution's shot cap
```

All three tool modules import from here instead of defining local copies.

### I3 — a coupling diagram that actually shows something useful, and actually renders

**`chatqec-mcp-tools` side (`_coupling_diagram.py`):** `render_coupling_diagram` currently
draws the entire `hw_graph`. Change it to draw only a **local neighborhood**: starting from
`highlighted_qubits` (the mapped physical qubits), do a breadth-first expansion outward
(1 hop by default) to include their immediate hardware neighbors, capped at a maximum node
count (e.g. 30 — generous enough to show real local structure, small enough to stay cheap
regardless of the underlying device's total qubit count). Render only that induced subgraph.
This makes the diagram naturally small on every backend size, and more genuinely useful (a
156-node hairball was never a good picture of the answer; "here are your qubits and what
they connect to" is). Node coloring (mapped = highlighted color, non-mapped neighbors =
default) stays as today; the existing try/finally figure-cleanup and best-effort
"never raises" contract are unchanged.

**`chatQEC` side, two changes:**

1. `chatqec_app/components/tool_output.py` gains a rendering branch for `ibm_map_circuit`,
   mirroring the existing `threshold_sweep` branch exactly (same `ast.literal_eval` parsing
   of `call.output`, same `ok`/`error_message` check, then `_render_svg(data.get("coupling_map_svg") or "")`
   if it looks like real SVG content). Today `ibm_map_circuit` falls through to the generic
   `st.code(out[:2000])` block, so its diagram is never shown — this closes that gap.
2. `src/chatqec/conversation/tool_loop.py`'s `_serialize_tool_result(output: dict) -> str`
   gains generic size-truncation: any string-valued field above a threshold (e.g. 4000
   characters) is replaced with a short placeholder (`"<omitted: N chars, rendered to the
   user separately>"`) before `json.dumps`, so a large field (this diagram, or any future
   tool's large output) doesn't get re-serialized into the model's conversation history on
   every remaining tool-loop iteration. This is a general safety net, not `ibm_map_circuit`-
   specific — it protects every current and future tool. The `trace("tool_loop.tool_result",
   ...)` call one line away logs the *pre-serialization* `result` dict directly; apply the
   same truncation there too (or route both through one shared helper) so trace logs don't
   grow unbounded either.

`prompts.py`'s tool list (already mentioning `stim_simulate`/`stim_diagram`/etc.) should also
list `ibm_map_circuit` for consistency — a one-line addition, not a design decision.

## Error handling

| Failure | Behavior |
|---|---|
| `_transpile` fails inside `ibm_map_circuit` (e.g. a placement qiskit's transpiler can't route within its own internal limits) | Already covered by `compute_mapping`'s existing broad `except Exception` around the post-placement compute block (added in the original feature's Task 11 fix) — returns `ok=False`, no new handling needed |
| Neighborhood-diagram BFS finds fewer than the cap (small backend) or the mapped qubits are already the whole graph | Renders whatever the neighborhood actually is; no special-casing needed, the cap is a ceiling not a target |
| `tool_output.py`'s `ibm_map_circuit` branch receives a malformed/non-dict `out` | Same `ast.literal_eval` failure handling already used by every other branch in that file (falls through to the generic code-block display) |
| A future tool emits an even-larger single field than today's SVGs | Caught by the same generic truncation, not a per-tool special case |

## Testing

- `_quality_model.py`: update/replace the existing `maximum_fidelity`-specific test
  (`test_maximum_fidelity_objective_uses_readout_only`) with one confirming T1/T2 now
  influence the score under `maximum_fidelity`, mirroring the existing "balanced" tests.
- `_transpile.py`: no new tests needed beyond what already exists for it in
  `submit_execution.py`'s test file, once relocated — same lazy-import pattern, same
  mocking-boundary convention as everywhere else in this package.
- `map_circuit.py`: new tests confirming `transpiled_depth`/`transpiled_two_qubit_gate_count`
  reflect a real transpiled circuit (mock `_transpile` to return a circuit with a known SWAP
  inserted, assert the reported count includes it) and that `estimated_fidelity` changes
  (goes down) for a placement requiring a SWAP versus one that doesn't.
- `preview_execution.py`: test `within_submit_caps` is `True`/`False` correctly for
  above/below-cap placements.
- `submit_execution.py`: test `objective` is passed through to `compute_mapping` (an identity
  or call-argument check, matching this project's established convention for this kind of
  wiring test).
- `_coupling_diagram.py`: test the neighborhood cap actually bounds SVG node count on a large
  synthetic graph (e.g. a 200-node graph, confirm the rendered node count stays at or under
  the cap regardless).
- `chatQEC` side: a `tool_output.py` test for the new `ibm_map_circuit` branch (matching
  the existing `threshold_sweep` test's shape), and a `tool_loop.py` test that
  `_serialize_tool_result` truncates a large field and that the untruncated original still
  reaches wherever `tool_output.py` reads from (the truncation must apply only to what goes
  into the model's context, never to what the UI renders from).

## Known limitations

- The neighborhood diagram intentionally no longer shows full-device context (e.g. "how far
  is this region from the edge of the chip"). This is an accepted tradeoff for making the
  diagram small and fast on every backend size; a future "show me the whole device" separate
  tool/flag could be added if a user ever asks for it, but isn't part of this fix.
- `estimated_fidelity`'s SWAP-aware recomputation still uses `_hardware_graph.py`'s real
  calibration data, so it's only as fresh as that data's own TTL cache — no new staleness
  concern beyond what already exists for `ibm_backend_calibration`.

## Out of scope

Actually implementing routing-aware objective search (choosing a placement that *minimizes*
SWAPs as a first-class objective, versus just reporting the SWAP cost after the fact) —
`_placement.py`'s existing greedy/VF2 strategies are unchanged by this spec. A configurable
neighborhood-hop-radius parameter exposed to the caller — starts fixed at 1 hop / 30-node cap,
tunable later if needed. Retrofitting `ibm_backend_calibration`'s own output format to also use
the shared `_limits.py` module (it doesn't currently duplicate any of these constants, so
there's nothing to consolidate there).
