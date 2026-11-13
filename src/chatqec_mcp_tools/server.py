"""MCP server entry - wires the two tools to the MCP `Server` class.

The real `mcp.server.Server` is created in `main()`; for testing we expose a
`build_server()` returning a lightweight registry wrapper.
"""

from __future__ import annotations

import io
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TextIO

from chatqec_mcp_tools.tools.code_params import (
    CodeParamsInput,
    code_params,
)
from chatqec_mcp_tools.tools.decoder_compare import (
    DecoderCompareInput,
    decoder_compare,
)
from chatqec_mcp_tools.tools.deltakit_memory import (
    DeltakitMemoryInput,
    deltakit_memory,
)
from chatqec_mcp_tools.tools.ft_primitive_simulate import (
    FtPrimitiveInput,
    ft_primitive_simulate,
)
from chatqec_mcp_tools.tools.glcb_visualize import (
    GlcbVisualizeInput,
    glcb_visualize_url,
)
from chatqec_mcp_tools.tools.guppy_run import GuppyRunInput, guppy_run
from chatqec_mcp_tools.tools.ibm_hardware.calibration import (
    IbmBackendCalibrationInput,
    ibm_backend_calibration,
)
from chatqec_mcp_tools.tools.ibm_hardware.ft_compile import (
    IbmFtCompileInput,
    ibm_ft_compile,
)
from chatqec_mcp_tools.tools.ibm_hardware.list_backends import (
    IbmListBackendsInput,
    ibm_list_backends,
)
from chatqec_mcp_tools.tools.ibm_hardware.map_circuit import (
    IbmMapCircuitInput,
    ibm_map_circuit,
)
from chatqec_mcp_tools.tools.ibm_hardware.pauli_checks import (
    IbmAddPauliChecksInput,
    ibm_add_pauli_checks,
)
from chatqec_mcp_tools.tools.ibm_hardware.preview_execution import (
    IbmPreviewExecutionInput,
    ibm_preview_execution,
)
from chatqec_mcp_tools.tools.ibm_hardware.submit_execution import (
    IbmSubmitExecutionInput,
    ibm_submit_execution,
)
from chatqec_mcp_tools.tools.iqm_hardware.calibration import (
    IqmBackendCalibrationInput,
    iqm_backend_calibration,
)
from chatqec_mcp_tools.tools.iqm_hardware.job_result import IqmJobResultInput, iqm_job_result
from chatqec_mcp_tools.tools.iqm_hardware.list_backends import (
    IqmListBackendsInput,
    iqm_list_backends,
)
from chatqec_mcp_tools.tools.iqm_hardware.map_circuit import IqmMapCircuitInput, iqm_map_circuit
from chatqec_mcp_tools.tools.iqm_hardware.preview_execution import (
    IqmPreviewExecutionInput,
    iqm_preview_execution,
)
from chatqec_mcp_tools.tools.iqm_hardware.submit_execution import (
    IqmSubmitExecutionInput,
    iqm_submit_execution,
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


def _ibm_add_pauli_checks_handler(args: dict[str, Any]) -> dict[str, Any]:
    return ibm_add_pauli_checks(IbmAddPauliChecksInput.model_validate(args)).model_dump()


def _ibm_ft_compile_handler(args: dict[str, Any]) -> dict[str, Any]:
    return ibm_ft_compile(IbmFtCompileInput.model_validate(args)).model_dump()


def _iqm_list_backends_handler(args: dict[str, Any]) -> dict[str, Any]:
    return iqm_list_backends(IqmListBackendsInput.model_validate(args)).model_dump()


def _iqm_backend_calibration_handler(args: dict[str, Any]) -> dict[str, Any]:
    return iqm_backend_calibration(IqmBackendCalibrationInput.model_validate(args)).model_dump()


def _iqm_map_circuit_handler(args: dict[str, Any]) -> dict[str, Any]:
    return iqm_map_circuit(IqmMapCircuitInput.model_validate(args)).model_dump()


def _iqm_preview_execution_handler(args: dict[str, Any]) -> dict[str, Any]:
    return iqm_preview_execution(IqmPreviewExecutionInput.model_validate(args)).model_dump()


def _iqm_submit_execution_handler(args: dict[str, Any]) -> dict[str, Any]:
    return iqm_submit_execution(IqmSubmitExecutionInput.model_validate(args)).model_dump()


def _iqm_job_result_handler(args: dict[str, Any]) -> dict[str, Any]:
    return iqm_job_result(IqmJobResultInput.model_validate(args)).model_dump()


def _deltakit_memory_handler(args: dict[str, Any]) -> dict[str, Any]:
    return deltakit_memory(DeltakitMemoryInput.model_validate(args)).model_dump()


def _ft_primitive_simulate_handler(args: dict[str, Any]) -> dict[str, Any]:
    return ft_primitive_simulate(FtPrimitiveInput.model_validate(args)).model_dump()


def _guppy_run_handler(args: dict[str, Any]) -> dict[str, Any]:
    return guppy_run(GuppyRunInput.model_validate(args)).model_dump()


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
            "Report a named QEC code's parameters. For surface_code_rotated, "
            "repetition_code and color_code: [[n, k]], stabilizer-generator count and the "
            "Stim-computed code distance at a given distance. For the literature families "
            "bb5 (bivariate bicycle, weight-5 checks), tile, directional_tile and barbell "
            "(IQM): the [[n, k, d]] codes reported in the papers, each with its arXiv source "
            "and table/section (not computed here); distance is optional and filters them. "
            "Always cite the returned source for these."
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
        name="ft_primitive_simulate",
        description=(
            "Simulate a fault-tolerant surface-code primitive with FTPrimitiveBench "
            "(arXiv:2605.04049): memory, transversal Hadamard, lattice surgery (MZZ/MXX) or a "
            "logical S gate, on a square or rectangular patch (x_distance, z_distance), under "
            "uniform, Pauli-biased (axis, bias_factor), measurement-biased or non-uniform "
            "(sigma) circuit-level noise. Samples with Stim and decodes with PyMatching. "
            "Returns the logical error rate with a 95% Wilson interval, the round structure, "
            "qubit/detector/observable counts and the circuit's SHA-256. Use it for biased "
            "noise, asymmetric patches, transversal H, lattice surgery or S-gate questions. "
            "Needs the `ftprimitive` extra."
        ),
        input_schema=FtPrimitiveInput.model_json_schema(),
        handler=_ft_primitive_simulate_handler,
    ),
    ToolSpec(
        name="deltakit_memory",
        description=(
            "Run a quantum-memory experiment with Riverlane's Deltakit: build a rotated or "
            "unrotated planar, toric or repetition code memory circuit, add circuit-level "
            "noise (ToyNoise: two-qubit error p and measurement flip), sample it and decode "
            "with Deltakit's local PyMatching decoder. Returns the logical error rate with "
            "its standard error plus qubit, detector and observable counts. Runs fully "
            "locally. Deltakit's other decoders (including the leakage-aware local "
            "clustering decoder) and leakage simulation are cloud services that are NOT "
            "configured here and are refused. Needs the `deltakit` extra."
        ),
        input_schema=DeltakitMemoryInput.model_json_schema(),
        handler=_deltakit_memory_handler,
    ),
    ToolSpec(
        name="guppy_run",
        description=(
            "Run a Guppy quantum program on the Selene emulator and return shot counts. "
            "Pass the program as source: guppylang imports and @guppy functions only, "
            "defining main() and reporting outcomes with result('name', value); e.g. a Bell "
            "pair gives only 0,0 and 1,1. With guppylang >= 1.0 a measurement must be read "
            "first, result('c0', measure(q).read()); older versions take measure(q) "
            "directly, and a compile error in stderr_tail says which. Counts are keyed by the register values joined by "
            "commas. The tool executes the submitted program and has no OS sandbox, so it is "
            "DISABLED unless the server is started with CHATQEC_ALLOW_GUPPY_EXEC=1; if it "
            "reports being disabled, tell the user that. A run takes about 15 seconds. "
            "Needs the `guppy` extra."
        ),
        input_schema=GuppyRunInput.model_json_schema(),
        handler=_guppy_run_handler,
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
        name="ibm_add_pauli_checks",
        description=(
            "Add spacetime Pauli checks (IBM's qiskit-paulice) to a Clifford circuit "
            "(OpenQASM 2 or Stim, ending in measurements) for error detection by "
            "postselection. Returns the checked circuit as OpenQASM and the overhead: "
            "qubits, gate count and depth before and after, Paulice's sampling cost, and, "
            "when noise_p is given, the fraction of shots postselection would discard. "
            "Non-Clifford circuits are rejected. Needs the `paulice` extra."
        ),
        input_schema=IbmAddPauliChecksInput.model_json_schema(),
        handler=_ibm_add_pauli_checks_handler,
    ),
    ToolSpec(
        name="ibm_ft_compile",
        description=(
            "Compile an OpenQASM 2 circuit for fault-tolerant execution with Qiskit's "
            "preset pass managers. mode=clifford_t approximates rotations with gridsynth "
            "into Clifford+T and reports the T count (gridsynth_precision is decimal "
            "digits: more digits, more T gates); mode=pbc converts to Pauli-based "
            "computation and reports the Pauli product rotation count. Very large "
            "compiled circuits are summarised by counts only. Needs Qiskit >= 2.5 (the "
            "`ft` extra)."
        ),
        input_schema=IbmFtCompileInput.model_json_schema(),
        handler=_ibm_ft_compile_handler,
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
            "coupling-map diagram. Free, local compute plus one calibration fetch. "
            "mapper='mapomatic' instead transpiles first and picks the lowest-error layout "
            "of the routed circuit (counts SWAP-added gates); default mapper places the "
            "logical circuit directly."
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
    ToolSpec(
        name="iqm_list_backends",
        description=(
            "List the IQM quantum computer behind the configured IQM server (ORNL): name, "
            "chip label, qubit and coupler count, health, native gates, calibration set. "
            "Free, cached; every answer built from this must state the data's age and "
            "flag it if stale=true. IQM does not report queue depth."
        ),
        input_schema=IqmListBackendsInput.model_json_schema(),
        handler=_iqm_list_backends_handler,
    ),
    ToolSpec(
        name="iqm_backend_calibration",
        description=(
            "Fetch the IQM device's current quality metrics: per-qubit T1, T2 echo, T2*, "
            "readout fidelity and errors, single-qubit (prx) RB fidelity; per-coupler CZ "
            "fidelity (interleaved RB), two-qubit Clifford fidelity and CZ duration. "
            "Qubits are named QB1, QB2, ... Cached; state the data's age in the answer."
        ),
        input_schema=IqmBackendCalibrationInput.model_json_schema(),
        handler=_iqm_backend_calibration_handler,
    ),
    ToolSpec(
        name="iqm_map_circuit",
        description=(
            "Map a circuit (QASM or a bell/ghz/qft/qaoa preset) onto the IQM device: "
            "physical qubits (QB names), strategy used, estimated fidelity, and a "
            "coupling-map diagram. Free, local compute plus one cached calibration fetch. "
            "mapper='mapomatic' instead transpiles first and picks the lowest-error layout "
            "of the routed circuit (counts SWAP-added gates)."
        ),
        input_schema=IqmMapCircuitInput.model_json_schema(),
        handler=_iqm_map_circuit_handler,
    ),
    ToolSpec(
        name="iqm_preview_execution",
        description=(
            "Preview a real IQM hardware run (mapped qubits, estimated fidelity, device "
            "health, shots) without submitting anything. Call this before "
            "iqm_submit_execution."
        ),
        input_schema=IqmPreviewExecutionInput.model_json_schema(),
        handler=_iqm_preview_execution_handler,
    ),
    ToolSpec(
        name="iqm_submit_execution",
        description=(
            "SUBMIT a real job to the IQM quantum computer; uses the account's hardware "
            "allocation. Only after the user has seen iqm_preview_execution and asked to "
            "run: set confirm_submit=true. Caps at 10 qubits / 4000 shots. Returns a job "
            "id; fetch counts later with iqm_job_result."
        ),
        input_schema=IqmSubmitExecutionInput.model_json_schema(),
        handler=_iqm_submit_execution_handler,
    ),
    ToolSpec(
        name="iqm_job_result",
        description=(
            "Status of an IQM job by job id and, once it is DONE, its measurement counts. Free."
        ),
        input_schema=IqmJobResultInput.model_json_schema(),
        handler=_iqm_job_result_handler,
    ),
]


def build_server() -> _TestServer:
    return _TestServer(_TOOLS)


def _isolate_protocol_stdout() -> TextIO:
    """Give the MCP protocol a private copy of stdout and point fd 1 at stderr.

    Native libraries (CUDA-Q, Gurobi, ...) write straight to fd 1. Over stdio
    MCP that line lands in the JSON-RPC stream, the client fails to parse it
    and closes the connection, so every later tool call in the session fails.
    After this call such output goes to stderr, which the client only logs.
    """
    sys.stdout.flush()
    proto_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return io.TextIOWrapper(os.fdopen(proto_fd, "wb"), encoding="utf-8", line_buffering=True)


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

    from pathlib import Path  # noqa: PLC0415

    from chatqec_mcp_tools.result_cache import (  # noqa: PLC0415
        cache_dir_from_env,
        cached_call,
        source_hash,
    )

    cache_dir = cache_dir_from_env()  # None unless CHATQEC_TOOL_CACHE_DIR is set
    code_hash = source_hash(Path(__file__).parent) if cache_dir else ""

    @app.call_tool()  # type: ignore[untyped-decorator]
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        for t in _TOOLS:
            if t.name == name:
                result = cached_call(
                    name, arguments, t.handler, cache_dir=cache_dir, code_hash=code_hash
                )
                return [TextContent(type="text", text=json.dumps(result))]
        raise ValueError(f"unknown tool: {name}")

    import anyio  # noqa: PLC0415

    protocol_out = anyio.wrap_file(_isolate_protocol_stdout())

    async def run() -> None:
        async with stdio_server(stdout=protocol_out) as (read, write):
            await app.run(read, write, app.create_initialization_options())

    asyncio.run(run())


if __name__ == "__main__":  # pragma: no cover
    main()
