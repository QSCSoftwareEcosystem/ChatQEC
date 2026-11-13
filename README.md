# chatqec-mcp-tools

MCP server exposing sandboxed Quantum Error Correction tools.

Tools:
- `stim_simulate` — run a Stim circuit; return measurement samples and detection events
- `stim_diagram` — render a Stim circuit as an SVG
- `pymatching_decode` — MWPM-decode a Stim circuit; return the logical error rate
- `threshold_sweep` — sweep p × distance for a code family; return a table + threshold plot (SVG)
- `code_params` — report a code's [[n,k]] + stabilizer count and a Stim-computed distance
- `qec_circuit_build` — build a stim.Circuit for a code family plain Stim doesn't ship
  built-in (toric code, color code, BB codes, PQRM), via the optional `lightstim` extra
- `tsim_simulate` — run a Tsim circuit (Stim syntax plus non-Clifford gates: T, T_DAG,
  R_X/R_Y/R_Z, U3, TPP/TPP_DAG, R_XX/R_YY/R_ZZ/R_PAULI, CCZ, CCX) and return
  detector/observable samples, via the optional `tsim` extra
- `glcb_visualize_url` — build a shareable QuEra Gemini Studio
  (bloqade.quera.com/studio/gemini, successor of the retired GLCB app) link for a small circuit;
  pure string encoding, no network call, no external dependency
- `ibm_list_backends` — list available IBM Quantum backends and their operational status,
  via the optional `ibm` extra (read-only; free)
- `ibm_backend_calibration` — fetch current calibration data (T1, T2, readout error, gate
  fidelities) for a backend, via the optional `ibm` extra (read-only; free)
- `ibm_map_circuit` — map and transpile a circuit onto an IBM backend's native gates and
  compute estimated fidelity, via the optional `ibm` extra (read-only; free)
- `ibm_preview_execution` — preview a circuit's mapped form and estimated fidelity on a
  backend without submitting, via the optional `ibm` extra (read-only; free)
- `ibm_submit_execution` — submit a circuit to an IBM backend for execution, via the
  optional `ibm` extra (spends real account credits; use only after `ibm_preview_execution`)

Superseded: `qiskit_qec_run` (code construction via qiskit-qec) — replaced by the
Stim-based `code_params`, since qiskit-qec is unshippable (not on PyPI; abandoned,
broken against modern Qiskit). Future: `stabilizer_check`, `code_search`,
`logical_error_curve`, `decode_compare`, `monte_carlo_estimate`.

## Install

Not published to PyPI — install straight from a clone:

```bash
git clone https://github.com/QSCSoftwareThrust/chatqec-mcp-tools
uv tool install ./chatqec-mcp-tools
```

Then point any MCP client (Claude Desktop, Claude Code, ChatQEC) at the `chatqec-mcp-tools` entry point.

`qec_circuit_build` additionally requires the optional `lightstim` extra:

```bash
uv tool install "./chatqec-mcp-tools[lightstim]"
# or, working inside the checkout: uv sync --extra lightstim
```

Without it installed, `qec_circuit_build` is still registered but always returns
`ok=False` with a message explaining the extra is required.

`tsim_simulate` additionally requires the optional `tsim` extra:

```bash
uv tool install "./chatqec-mcp-tools[tsim]"
# or, working inside the checkout: uv sync --extra tsim
```

Without it installed, `tsim_simulate` is still registered but always returns
`ok=False` with a message explaining the extra is required.

`glcb_visualize_url` has zero dependencies beyond this package itself (pure string
encoding, no network call) — no extra required.

The 5 IBM Quantum hardware-aware tools additionally require the optional `ibm` extra:

```bash
uv tool install "./chatqec-mcp-tools[ibm]"
# or, working inside the checkout: uv sync --extra ibm
```

Without it installed, the IBM tools are still registered but return `ok=False` with a
message explaining the extra is required.

The IBM tools need these environment variables:
- `QISKIT_IBM_TOKEN`: IBM Quantum access token (required for all IBM tools)
- `QISKIT_IBM_INSTANCE`: IBM Quantum instance identifier (optional; usually not needed
  on a free/open-plan account — the SDK auto-selects your account's single instance)

The 5 IBM tools are:
- `ibm_list_backends` — list available IBM Quantum backends and their operational status
  (read-only; free)
- `ibm_backend_calibration` — fetch current calibration data (T1, T2, readout error, gate
  fidelities) for a backend (read-only; free)
- `ibm_map_circuit` — map and transpile a circuit onto an IBM backend's native gates,
  compute estimated fidelity via quality model (read-only; free)
- `ibm_preview_execution` — preview a circuit's mapped form and estimated fidelity on a
  backend without submitting (read-only; free)
- `ibm_submit_execution` — submit a circuit to an IBM backend for execution. **WARNING:**
  This tool spends real account credits. Use only after reviewing the output of
  `ibm_preview_execution`; it is a soft safety guarantee that `ibm_submit_execution`
  should only be reached after a preview step.

## License

Apache-2.0.
