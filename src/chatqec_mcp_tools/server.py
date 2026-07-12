"""MCP server entry - wires the two tools to the MCP `Server` class.

The real `mcp.server.Server` is created in `main()`; for testing we expose a
`build_server()` returning a lightweight registry wrapper.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from chatqec_mcp_tools.tools.code_params import (
    CodeParamsInput,
    code_params,
)
from chatqec_mcp_tools.tools.pymatching_decode import (
    PymatchingDecodeInput,
    pymatching_decode,
)
from chatqec_mcp_tools.tools.stim_diagram import (
    StimDiagramInput,
    stim_diagram,
)
from chatqec_mcp_tools.tools.stim_simulate import (
    StimSimulateInput,
    stim_simulate,
)
from chatqec_mcp_tools.tools.threshold_sweep import (
    ThresholdSweepInput,
    threshold_sweep,
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]


def _stim_simulate_handler(args: dict[str, Any]) -> dict[str, Any]:
    return stim_simulate(StimSimulateInput.model_validate(args)).model_dump()


def _stim_diagram_handler(args: dict[str, Any]) -> dict[str, Any]:
    return stim_diagram(StimDiagramInput.model_validate(args)).model_dump()


def _pymatching_decode_handler(args: dict[str, Any]) -> dict[str, Any]:
    return pymatching_decode(PymatchingDecodeInput.model_validate(args)).model_dump()


def _threshold_sweep_handler(args: dict[str, Any]) -> dict[str, Any]:
    return threshold_sweep(ThresholdSweepInput.model_validate(args)).model_dump()


def _code_params_handler(args: dict[str, Any]) -> dict[str, Any]:
    return code_params(CodeParamsInput.model_validate(args)).model_dump()


class _TestServer:
    """Lightweight server registry used in unit tests."""

    def __init__(self, tools: list[ToolSpec]) -> None:
        self._tools = {t.name: t for t in tools}

    def list_tools(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def invoke_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return self._tools[name].handler(args)


_TOOLS = [
    ToolSpec(
        name="stim_simulate",
        description="Run a Stim circuit and return measurement samples.",
        input_schema=StimSimulateInput.model_json_schema(),
        handler=_stim_simulate_handler,
    ),
    ToolSpec(
        name="stim_diagram",
        description="Render a Stim circuit as an SVG diagram.",
        input_schema=StimDiagramInput.model_json_schema(),
        handler=_stim_diagram_handler,
    ),
    ToolSpec(
        name="pymatching_decode",
        description=(
            "MWPM-decode a Stim circuit (with detectors + observables) and return "
            "the logical error rate over N shots."
        ),
        input_schema=PymatchingDecodeInput.model_json_schema(),
        handler=_pymatching_decode_handler,
    ),
    ToolSpec(
        name="threshold_sweep",
        description=(
            "Sweep physical error rate p across code distances for a code family "
            "(surface_code_rotated, repetition_code, color_code); MWPM-decode each "
            "and return a logical-error-rate table plus a log-log threshold plot (SVG)."
        ),
        input_schema=ThresholdSweepInput.model_json_schema(),
        handler=_threshold_sweep_handler,
    ),
    ToolSpec(
        name="code_params",
        description=(
            "Report a named QEC code's parameters — [[n, k]], stabilizer-generator "
            "count, and the Stim-computed code distance — for a family "
            "(surface_code_rotated, repetition_code, color_code) at a given distance."
        ),
        input_schema=CodeParamsInput.model_json_schema(),
        handler=_code_params_handler,
    ),
]


def build_server() -> _TestServer:
    return _TestServer(_TOOLS)


def main() -> None:  # pragma: no cover
    """Real MCP entry - wires _TOOLS to mcp.server.Server over stdio."""
    import asyncio  # noqa: PLC0415
    import json  # noqa: PLC0415

    from mcp.server import Server  # noqa: PLC0415
    from mcp.server.stdio import stdio_server  # noqa: PLC0415
    from mcp.types import TextContent, Tool  # noqa: PLC0415

    app: Server = Server("chatqec-mcp-tools")

    @app.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
    async def list_tools() -> list[Tool]:
        return [
            Tool(name=t.name, description=t.description, inputSchema=t.input_schema)
            for t in _TOOLS
        ]

    @app.call_tool()  # type: ignore[untyped-decorator]
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        for t in _TOOLS:
            if t.name == name:
                result = t.handler(arguments)
                return [TextContent(type="text", text=json.dumps(result))]
        raise ValueError(f"unknown tool: {name}")

    async def run() -> None:
        async with stdio_server() as (read, write):
            await app.run(read, write, app.create_initialization_options())

    asyncio.run(run())


if __name__ == "__main__":  # pragma: no cover
    main()
