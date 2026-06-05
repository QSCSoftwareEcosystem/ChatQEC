# ChatQEC

Open-source semi-automated research assistant for Quantum Error Correction.

ChatQEC answers QEC questions over a curated corpus of research papers, textbooks,
tutorials, code repositories, and video transcripts, with inline citations and
sandboxed Stim execution for circuit-level questions.

## Status

Pre-alpha. See [the design spec](docs/superpowers/specs/2026-05-29-chatqec-design.md)
and [the implementation plan](docs/superpowers/plans/2026-05-29-chatqec-implementation.md).

## Quick start (developers)

```bash
uv venv
uv sync --extra dev
uv run chatqec --help
uv run pytest
```

## Stack

Python 3.11+, uv, Hydra, Qdrant, Anthropic + Gemini SDKs, Streamlit. Sandboxed
QEC tooling (Stim, PyMatching, qiskit-qec) lives in a sibling
[`chatqec-mcp-tools`](https://code.ornl.gov/qsc-as/chatqec-mcp-tools) MCP server.

## License

Apache 2.0 — see [LICENSE](LICENSE).
