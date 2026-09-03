# src/chatqec_mcp_tools/tools/ibm_hardware/map_circuit.py
"""ibm_map_circuit tool -- map a circuit onto a chosen or auto-selected IBM
backend using the ported quality-scored placement engine."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from chatqec_mcp_tools.tools.ibm_hardware._circuits import (
    build_interaction_graph,
    parse_circuit_input,
)
from chatqec_mcp_tools.tools.ibm_hardware._client import IBMCredentialsError, get_ibm_service
from chatqec_mcp_tools.tools.ibm_hardware._coupling_diagram import render_coupling_diagram
from chatqec_mcp_tools.tools.ibm_hardware._hardware_graph import build_hardware_graph
from chatqec_mcp_tools.tools.ibm_hardware._placement import PlacementResult, place
from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel

_VALID_OBJECTIVES = ("maximum_fidelity", "minimum_execution_time")
_TWO_QUBIT_GATE_ARITY = 2


class IbmMapCircuitInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    circuit_qasm: str | None = None
    circuit_preset: str | None = None
    num_qubits: int | None = None
    qaoa_layers: int | None = None
    backend: str | None = None
    objective: str = "maximum_fidelity"


class IbmMapCircuitOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    backend: str = ""
    physical_qubits: list[int] = Field(default_factory=list)
    logical_to_physical: dict[str, int] = Field(default_factory=dict)
    strategy_used: str = ""
    transpiled_two_qubit_gate_count: int = 0
    estimated_fidelity: float = 0.0
    coupling_map_svg: str = ""
    fetched_at: str = ""
    age_seconds: int = 0
    error_message: str = ""


@dataclass
class MappingOutcome:
    """Shared result shape reused by ibm_preview_execution / ibm_submit_execution
    so they don't duplicate this tool's mapping logic."""

    ok: bool
    backend: str = ""
    placement: PlacementResult | None = None
    two_qubit_gate_count: int = 0
    estimated_fidelity: float = 0.0
    svg: str = ""
    error_message: str = ""
    service: object = None  # the authenticated service, reused by callers so they
    sdk_backend: object = None  # don't re-authenticate / re-fetch the backend a second time


def _auto_select_backend(service: Any, *, min_qubits: int) -> Any | None:
    """Pick the operational backend with the fewest pending jobs among those
    with enough qubits; ties broken by name for determinism. Returns None if
    no backend qualifies."""
    candidates: list[tuple[int, str, Any]] = []
    for backend in service.backends():
        if backend.num_qubits < min_qubits:
            continue
        status = backend.status()
        if not status.operational:
            continue
        candidates.append((status.pending_jobs, backend.name, backend))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def compute_mapping(  # noqa: PLR0911 -- each early return is a distinct validation/error path
    *,
    circuit_qasm: str | None,
    circuit_preset: str | None,
    num_qubits: int | None,
    qaoa_layers: int | None,
    backend: str | None,
    objective: str,
    include_svg: bool = True,
) -> MappingOutcome:
    if objective not in _VALID_OBJECTIVES:
        return MappingOutcome(ok=False, error_message=f"objective must be one of {_VALID_OBJECTIVES}")

    try:
        spec = parse_circuit_input(
            circuit_qasm=circuit_qasm,
            circuit_preset=circuit_preset,
            num_qubits=num_qubits,
            qaoa_layers=qaoa_layers,
        )
    except ValueError as exc:
        return MappingOutcome(ok=False, error_message=str(exc))

    try:
        service = get_ibm_service()
        if backend:
            sdk_backend = service.backend(backend)
        else:
            sdk_backend = _auto_select_backend(service, min_qubits=spec.num_logical_qubits)
            if sdk_backend is None:
                return MappingOutcome(
                    ok=False,
                    error_message=(
                        f"No operational backend with at least {spec.num_logical_qubits} "
                        "qubits is visible to this account."
                    ),
                )
        hw_graph = build_hardware_graph(sdk_backend)
    except IBMCredentialsError as exc:
        return MappingOutcome(ok=False, error_message=str(exc))
    except Exception as exc:
        return MappingOutcome(ok=False, error_message=f"Failed to load backend {backend}: {exc}")

    backend = sdk_backend.name

    hw_num_qubits = hw_graph.graph.number_of_nodes()
    if spec.num_logical_qubits > hw_num_qubits:
        return MappingOutcome(
            ok=False,
            error_message=(
                f"Circuit needs {spec.num_logical_qubits} qubits but {backend} "
                f"has only {hw_num_qubits}."
            ),
        )

    interaction = build_interaction_graph(spec.circuit)
    quality = HardwareQualityModel(hw_graph.graph, objective=objective)
    placement = place(hw_graph.graph, interaction.graph, quality)

    two_q_count = sum(1 for instr in spec.circuit.data if len(instr.qubits) == _TWO_QUBIT_GATE_ARITY)
    fidelities = [
        quality.edge_score(placement.assignment[u], placement.assignment[v])
        for u, v in interaction.graph.edges
        if u in placement.assignment and v in placement.assignment
    ]
    estimated_fidelity = sum(fidelities) / len(fidelities) if fidelities else 1.0

    svg = ""
    if include_svg:
        svg = render_coupling_diagram(hw_graph.graph, set(placement.assignment.values()))

    return MappingOutcome(
        ok=True,
        backend=backend,
        placement=placement,
        two_qubit_gate_count=two_q_count,
        estimated_fidelity=estimated_fidelity,
        svg=svg,
        service=service,
        sdk_backend=sdk_backend,
    )


def ibm_map_circuit(inp: IbmMapCircuitInput) -> IbmMapCircuitOutput:
    outcome = compute_mapping(
        circuit_qasm=inp.circuit_qasm,
        circuit_preset=inp.circuit_preset,
        num_qubits=inp.num_qubits,
        qaoa_layers=inp.qaoa_layers,
        backend=inp.backend,
        objective=inp.objective,
    )
    if not outcome.ok or outcome.placement is None:
        return IbmMapCircuitOutput(ok=False, error_message=outcome.error_message)

    return IbmMapCircuitOutput(
        ok=True,
        backend=outcome.backend,
        physical_qubits=sorted(outcome.placement.assignment.values()),
        logical_to_physical={str(k): v for k, v in outcome.placement.assignment.items()},
        strategy_used=outcome.placement.strategy_used,
        transpiled_two_qubit_gate_count=outcome.two_qubit_gate_count,
        estimated_fidelity=outcome.estimated_fidelity,
        coupling_map_svg=outcome.svg,
        fetched_at=dt.datetime.now(tz=dt.UTC).isoformat(),
        age_seconds=0,
    )


__all__ = [
    "IbmMapCircuitInput",
    "IbmMapCircuitOutput",
    "MappingOutcome",
    "compute_mapping",
    "ibm_map_circuit",
]
