# src/chatqec_mcp_tools/tools/ibm_hardware/map_circuit.py
"""ibm_map_circuit tool -- map a circuit onto a chosen or auto-selected IBM
backend using the ported quality-scored placement engine."""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from chatqec_mcp_tools.tools.ibm_hardware._cache import get_or_fetch
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
_MAX_NUM_QUBITS = 1024
# Same exclusion set _circuits.build_interaction_graph uses -- these gates never
# count as a "two-qubit gate" even when they happen to take two qubit operands.
_NON_INTERACTION_GATE_NAMES = frozenset({"barrier", "measure", "reset", "delay"})
# Shared with list_backends.py's cache key/TTL convention so ibm_map_circuit's
# auto-selection and ibm_list_backends read/write the same cache entry instead
# of each hitting the IBM API independently.
_BACKEND_LIST_CACHE_KEY = "list_backends"
_BACKEND_LIST_DEFAULT_TTL_SECONDS = 60.0


class IbmMapCircuitInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    circuit_qasm: str | None = None
    circuit_preset: str | None = None
    num_qubits: int | None = Field(default=None, ge=1, le=_MAX_NUM_QUBITS)
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
    service: Any = None  # the authenticated service, reused by callers so they
    sdk_backend: Any = None  # don't re-authenticate / re-fetch the backend a second time


def _fetch_backend_infos(service: Any) -> list[dict[str, Any]]:
    """Same shape as list_backends.py's own fetch -- kept format-compatible
    since both tools read/write the same cache entry (see
    _BACKEND_LIST_CACHE_KEY)."""
    infos = []
    for backend in service.backends():
        status = backend.status()
        infos.append(
            {
                "name": backend.name,
                "num_qubits": backend.num_qubits,
                "operational": status.operational,
                "pending_jobs": status.pending_jobs,
                "basis_gates": list(backend.operation_names),
            }
        )
    return infos


def _auto_select_backend(service: Any, *, min_qubits: int) -> Any | None:
    """Pick the operational backend with the fewest pending jobs among those
    with enough qubits; ties broken by name for determinism. Returns None if
    no backend qualifies. The backend list itself is TTL-cached (shared with
    ibm_list_backends) rather than re-fetched on every mapping call."""
    ttl = float(os.environ.get("CHATQEC_IBM_BACKEND_LIST_TTL_SECONDS", _BACKEND_LIST_DEFAULT_TTL_SECONDS))
    infos = get_or_fetch(_BACKEND_LIST_CACHE_KEY, ttl, lambda: _fetch_backend_infos(service)).payload

    candidates: list[tuple[int, str]] = []
    for info in infos:
        if info["num_qubits"] < min_qubits:
            continue
        if not info["operational"]:
            continue
        candidates.append((info["pending_jobs"], info["name"]))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    chosen_name = candidates[0][1]
    return service.backend(chosen_name)


def compute_mapping(  # noqa: PLR0911, PLR0912 -- distinct validation/error paths, not complexity
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
    except (ValueError, OverflowError, ArithmeticError) as exc:
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

    try:
        interaction = build_interaction_graph(spec.circuit)
        quality = HardwareQualityModel(hw_graph.graph, objective=objective)
        placement = place(hw_graph.graph, interaction.graph, quality)

        two_q_count = sum(
            1
            for instr in spec.circuit.data
            if len(instr.qubits) == _TWO_QUBIT_GATE_ARITY
            and instr.operation.name not in _NON_INTERACTION_GATE_NAMES
        )

        # Genuine physical fidelity estimate: product of independent
        # per-component fidelities (each two-qubit gate actually used, plus
        # each mapped qubit's readout), not a min-max score normalized
        # against this backend's own best coupler (which would always credit
        # the single best coupler on any device with 1.0, however bad the
        # device is overall).
        estimated_fidelity = 1.0
        for u, v in interaction.graph.edges:
            pu, pv = placement.assignment.get(u), placement.assignment.get(v)
            if pu is not None and pv is not None and hw_graph.graph.has_edge(pu, pv):
                gate_fidelity = hw_graph.graph.edges[pu, pv].get("gate_fidelity")
                if gate_fidelity is not None:
                    estimated_fidelity *= gate_fidelity
        for physical_qubit in placement.assignment.values():
            readout_fidelity = hw_graph.graph.nodes[physical_qubit].get("readout_fidelity")
            if readout_fidelity is not None:
                estimated_fidelity *= readout_fidelity

        svg = ""
        if include_svg:
            svg = render_coupling_diagram(hw_graph.graph, set(placement.assignment.values()))
    except Exception as exc:
        return MappingOutcome(ok=False, error_message=f"Placement failed: {exc}")

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
