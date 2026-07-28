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

Superseded: `qiskit_qec_run` (code construction via qiskit-qec) — replaced by the
Stim-based `code_params`, since qiskit-qec is unshippable (not on PyPI; abandoned,
broken against modern Qiskit). Future: `stabilizer_check`, `code_search`,
`logical_error_curve`, `decode_compare`, `monte_carlo_estimate`.

## Install

```bash
uv tool install chatqec-mcp-tools
```

Then point any MCP client (Claude Desktop, Claude Code, ChatQEC) at the `chatqec-mcp-tools` entry point.

`qec_circuit_build` additionally requires the optional `lightstim` extra:

```bash
pip install "chatqec-mcp-tools[lightstim]"
# or: uv sync --extra lightstim
```

Without it installed, `qec_circuit_build` is still registered but always returns
`ok=False` with a message explaining the extra is required.

## License

Apache-2.0.
