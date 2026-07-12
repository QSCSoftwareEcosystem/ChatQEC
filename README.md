# chatqec-mcp-tools

MCP server exposing sandboxed Quantum Error Correction tools.

Tools:
- `stim_simulate` — run a Stim circuit; return measurement samples and detection events
- `stim_diagram` — render a Stim circuit as an SVG
- `pymatching_decode` — MWPM-decode a Stim circuit; return the logical error rate
- `threshold_sweep` — sweep p × distance for a code family; return a table + threshold plot (SVG)

Deferred: `qiskit_qec_run` (code construction + [[n,k,d]]) — blocked on qiskit-qec packaging
(not on PyPI; GitHub-only, pre-1.0, Python 3.7–3.10). Future: `stabilizer_check`,
`code_search`, `logical_error_curve`, `decode_compare`, `monte_carlo_estimate`.

## Install

```bash
uv tool install chatqec-mcp-tools
```

Then point any MCP client (Claude Desktop, Claude Code, ChatQEC) at the `chatqec-mcp-tools` entry point.

## License

Apache-2.0.
