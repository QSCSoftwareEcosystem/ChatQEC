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
from chatqec_mcp_tools.tools.decoder_compare import (
    DecoderCompareInput,
    decoder_compare,
)
from chatqec_mcp_tools.tools.glcb_visualize import (
    GlcbVisualizeInput,
    glcb_visualize_url,
)
from chatqec_mcp_tools.tools.ibm_hardware.calibration import (
    IbmBackendCalibrationInput,
    ibm_backend_calibration,
)
from chatqec_mcp_tools.tools.ibm_hardware.list_backends import (
    IbmListBackendsInput,
    ibm_list_backends,
)
from chatqec_mcp_tools.tools.ibm_hardware.map_circuit import (
    IbmMapCircuitInput,
    ibm_map_circuit,
)
from chatqec_mcp_tools.tools.ibm_hardware.preview_execution import (
    IbmPreviewExecutionInput,
    ibm_preview_execution,
)
from chatqec_mcp_tools.tools.ibm_hardware.submit_execution import (
    IbmSubmitExecutionInput,
    ibm_submit_execution,
)
from chatqec_mcp_tools.tools.pymatching_decode import (
    PymatchingDecodeInput,
    pymatching_decode,
)
from chatqec_mcp_tools.tools.qec_circuit_build import (
    QecCircuitBuildInput,
    qec_circuit_build,
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
from chatqec_mcp_tools.tools.tsim_simulate import (
    TsimSimulateInput,
    tsim_simulate,
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


def _qec_circuit_build_handler(args: dict[str, Any]) -> dict[str, Any]:
    return qec_circuit_build(QecCircuitBuildInput.model_validate(args)).model_dump()


def _tsim_simulate_handler(args: dict[str, Any]) -> dict[str, Any]:
    return tsim_simulate(TsimSimulateInput.model_validate(args)).model_dump()


def _decoder_compare_handler(args: dict[str, Any]) -> dict[str, Any]:
    return decoder_compare(DecoderCompareInput.model_validate(args)).model_dump()


def _glcb_visualize_url_handler(args: dict[str, Any]) -> dict[str, Any]:
    return glcb_visualize_url(GlcbVisualizeInput.model_validate(args)).model_dump()


def _ibm_list_backends_handler(args: dict[str, Any]) -> dict[str, Any]:
    return ibm_list_backends(IbmListBackendsInput.model_validate(args)).model_dump()


def _ibm_backend_calibration_handler(args: dict[str, Any]) -> dict[str, Any]:
    return ibm_backend_calibration(IbmBackendCalibrationInput.model_validate(args)).model_dump()


def _ibm_map_circuit_handler(args: dict[str, Any]) -> dict[str, Any]:
    return ibm_map_circuit(IbmMapCircuitInput.model_validate(args)).model_dump()


def _ibm_preview_execution_handler(args: dict[str, Any]) -> dict[str, Any]:
    return ibm_preview_execution(IbmPreviewExecutionInput.model_validate(args)).model_dump()


def _ibm_submit_execution_handler(args: dict[str, Any]) -> dict[str, Any]:
    return ibm_submit_execution(IbmSubmitExecutionInput.model_validate(args)).model_dump()


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
    ToolSpec(
        name="qec_circuit_build",
        description=(
            "Build a Stim circuit for a QEC code family beyond Stim's built-ins — "
            "toric code, color code, bivariate-bicycle (BB) codes, or PQRM — by "
            "delegating to LightStim (optional `lightstim` extra); returns ok=False "
            "with an error_message if lightstim isn't installed or the requested "
            "distance/rounds/protocol combination isn't supported."
        ),
        input_schema=QecCircuitBuildInput.model_json_schema(),
        handler=_qec_circuit_build_handler,
    ),
    ToolSpec(
        name="tsim_simulate",
        description=(
            "Run a Tsim circuit (Stim syntax plus non-Clifford gates — T, T_DAG, "
            "R_X/R_Y/R_Z, U3, TPP/TPP_DAG, R_XX/R_YY/R_ZZ/R_PAULI, CCZ, CCX) and "
            "return detector/observable samples (not raw measurement bits — the "
            "circuit must contain a DETECTOR or OBSERVABLE_INCLUDE instruction); "
            "enables magic-state distillation/cultivation circuits that plain Stim "
            "cannot simulate at all. Optional `tsim` extra — returns ok=False with "
            "an error_message if not installed."
        ),
        input_schema=TsimSimulateInput.model_json_schema(),
        handler=_tsim_simulate_handler,
    ),
    ToolSpec(
        name="decoder_compare",
        description=(
            "Compare QEC decoders on identical syndrome samples and rank them by "
            "logical error rate (with std error and decode time): pymatching (MWPM), "
            "tesseract (Google's search-based most-likely-error decoder, strong on "
            "color codes), bposd / bplsd / belief_find (ldpc qLDPC decoders via QuEra "
            "bloqade-decoders), mwpf (hypergraph MWPF), mle (exact most-likely-error via "
            "Gurobi, small codes, `mle` extra), table (lookup table, small codes), "
            "cudaq_relay_bp / cudaq_bposd / cudaq_sliding_window (NVIDIA CUDA-Q QEC GPU "
            "RelayBP, BP+OSD and sliding-window decoding; sliding window takes window_size "
            "and window_step). Pass a Stim circuit, or code_family + distance + p to "
            "generate a memory experiment. Non-MWPM decoders need the `decoders`, `mle` or "
            "`cudaq` extra; missing ones report ok=False per decoder."
        ),
        input_schema=DecoderCompareInput.model_json_schema(),
        handler=_decoder_compare_handler,
    ),
    ToolSpec(
        name="glcb_visualize_url",
        description=(
            "Build a shareable QuEra GLCB (gemini-logical-circuit-builder.vercel.app) "
            "visualization link for a small circuit. Returns a clickable URL with "
            "circuit diagram, expected values, noise sampling, decoding, and Bloqade "
            "code export; pure string encoding, no network call or external dependency."
        ),
        input_schema=GlcbVisualizeInput.model_json_schema(),
        handler=_glcb_visualize_url_handler,
    ),
    ToolSpec(
        name="ibm_list_backends",
        description=(
            "List IBM Quantum backends visible to the configured account: name, "
            "qubit count, operational status, and pending job queue depth. "
            "Free, cached for 60 seconds; every answer built from this must state "
            "the data's age (e.g. 'fetched 12 min ago') and flag it if stale=true."
        ),
        input_schema=IbmListBackendsInput.model_json_schema(),
        handler=_ibm_list_backends_handler,
    ),
    ToolSpec(
        name="ibm_backend_calibration",
        description=(
            "Fetch T1/T2/readout error and per-coupler two-qubit gate error/duration "
            "for one IBM backend. Cached; every answer built from this must state the "
            "data's age (e.g. 'fetched 12 min ago')."
        ),
        input_schema=IbmBackendCalibrationInput.model_json_schema(),
        handler=_ibm_backend_calibration_handler,
    ),
    ToolSpec(
        name="ibm_map_circuit",
        description=(
            "Map a circuit (QASM or a bell/ghz/qft/qaoa preset) onto an IBM backend: "
            "physical qubit layout, strategy used, estimated fidelity, and a "
            "coupling-map diagram. Free, local compute plus one calibration fetch."
        ),
        input_schema=IbmMapCircuitInput.model_json_schema(),
        handler=_ibm_map_circuit_handler,
    ),
    ToolSpec(
        name="ibm_preview_execution",
        description=(
            "Preview a real-hardware submission — mapped qubits, backend status, "
            "shots — without submitting or spending credits. Call this before "
            "ibm_submit_execution."
        ),
        input_schema=IbmPreviewExecutionInput.model_json_schema(),
        handler=_ibm_preview_execution_handler,
    ),
    ToolSpec(
        name="ibm_submit_execution",
        description=(
            "SUBMIT a real job to IBM Quantum hardware — spends account credits/quota. "
            "Requires an explicit backend name (no auto-pick), caps at 10 qubits / "
            "4000 shots. Call ibm_preview_execution first to see the plan before "
            "spending anything."
        ),
        input_schema=IbmSubmitExecutionInput.model_json_schema(),
        handler=_ibm_submit_execution_handler,
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
            Tool(name=t.name, description=t.description, inputSchema=t.input_schema) for t in _TOOLS
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
