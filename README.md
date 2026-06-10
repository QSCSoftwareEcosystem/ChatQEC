# chatqec-mcp-tools

MCP server exposing sandboxed Quantum Error Correction tools.

Tools (v1):
- `stim_simulate` — run a Stim circuit; return measurement samples and detection events
- `stim_diagram` — render a Stim circuit as an SVG

Future (v2): `pymatching_decode`, `qiskit_qec_run`, `threshold_sweep`, `stabilizer_check`, `code_search`, `logical_error_curve`, `decode_compare`, `monte_carlo_estimate`.

## Install

```bash
uv tool install chatqec-mcp-tools
```

Then point any MCP client (Claude Desktop, Claude Code, ChatQEC) at the `chatqec-mcp-tools` entry point.

## License

Apache-2.0.
