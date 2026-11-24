"""Command-line entry for the QEC tools: one process per call, JSON in and out.

    chatqec-tools list                 -> JSON list of {name, description, input_schema}
    chatqec-tools call <name> < args   -> JSON result of that tool (args JSON on stdin)

ChatQEC runs ``call`` in a fresh subprocess per tool call, so a timeout is a plain
process kill and a native crash ends only that call. The MCP server in
``server.py`` stays for MCP clients such as Claude Desktop.

Exit codes: 0 with a JSON object on stdout (a failed tool returns ``ok: false``),
2 for usage errors (unknown tool, arguments that are not a JSON object).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO

from chatqec_mcp_tools.result_cache import cache_dir_from_env, cached_call, source_hash
from chatqec_mcp_tools.server import _TOOLS, _isolate_protocol_stdout

EXIT_USAGE = 2


def _list(out: TextIO) -> int:
    tools = [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in _TOOLS
    ]
    json.dump(tools, out)
    out.write("\n")
    return 0


def _read_args(raw: str) -> dict[str, Any] | None:
    if not raw.strip():
        return {}
    try:
        args = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return args if isinstance(args, dict) else None


def _call(name: str, raw_args: str, out: TextIO) -> int:
    spec = next((t for t in _TOOLS if t.name == name), None)
    if spec is None:
        print(f"unknown tool: {name}", file=sys.stderr)
        return EXIT_USAGE
    args = _read_args(raw_args)
    if args is None:
        print("arguments must be a JSON object on stdin", file=sys.stderr)
        return EXIT_USAGE
    cache_dir = cache_dir_from_env()  # None unless CHATQEC_TOOL_CACHE_DIR is set
    code_hash = source_hash(Path(__file__).parent) if cache_dir else ""
    try:
        result = cached_call(name, args, spec.handler, cache_dir=cache_dir, code_hash=code_hash)
    except Exception as exc:  # same contract as an MCP isError result
        result = {"ok": False, "error_message": f"{type(exc).__name__}: {exc}"}
    json.dump(result, out, default=str)
    out.write("\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chatqec-tools", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="print every tool as JSON")
    call = sub.add_parser("call", help="run one tool; arguments are a JSON object on stdin")
    call.add_argument("name")
    ns = parser.parse_args(argv)

    # Native libraries print straight to fd 1; keep stdout for the JSON result only.
    out = _isolate_protocol_stdout()
    if ns.command == "list":
        return _list(out)
    return _call(ns.name, sys.stdin.read(), out)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
