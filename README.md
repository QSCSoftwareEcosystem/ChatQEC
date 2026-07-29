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
- `glcb_visualize_url` — build a shareable QuEra GLCB
  (gemini-logical-circuit-builder.vercel.app) visualization link for a small circuit;
  pure string encoding, no network call, no external dependency

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

## License

Apache-2.0.
