"""iqm_map_circuit tool -- place a circuit on the IQM device with the same quality-scored
placement engine the IBM tools use (ibm_hardware._placement / _quality_model)."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from chatqec_mcp_tools.tools.ibm_hardware._circuits import (
    build_interaction_graph,
    parse_circuit_input,
)
from chatqec_mcp_tools.tools.ibm_hardware._client import IBM_EXTRA_INSTALL_HINT
from chatqec_mcp_tools.tools.ibm_hardware._coupling_diagram import render_coupling_diagram
from chatqec_mcp_tools.tools.ibm_hardware._placement import PlacementResult, place
from chatqec_mcp_tools.tools.ibm_hardware._quality_model import HardwareQualityModel
from chatqec_mcp_tools.tools.iqm_hardware._client import IQMCredentialsError
from chatqec_mcp_tools.tools.iqm_hardware._hardware_graph import (
    build_hardware_graph,
    get_snapshot,
    named_graph,
)

_VALID_OBJECTIVES = ("maximum_fidelity", "minimum_execution_time")
_MAX_NUM_QUBITS = 1024
_MAX_QAOA_LAYERS = 1000
_NON_INTERACTION_GATE_NAMES = frozenset({"barrier", "measure", "reset", "delay"})


class IqmMapCircuitInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    circuit_qasm: str | None = None
    circuit_preset: str | None = None
    num_qubits: int | None = Field(default=None, ge=1, le=_MAX_NUM_QUBITS)
    qaoa_layers: int | None = Field(default=None, ge=1, le=_MAX_QAOA_LAYERS)
    objective: str = "maximum_fidelity"


class IqmMapCircuitOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    backend: str = ""
    dut_label: str = ""
    physical_qubits: list[str] = Field(default_factory=list)
    logical_to_physical: dict[str, str] = Field(default_factory=dict)
    strategy_used: str = ""
    two_qubit_gate_count: int = 0
    estimated_fidelity: float = 0.0
    coupling_map_svg: str = ""
    fetched_at: str = ""
    age_seconds: int = 0
    stale: bool = False
    error_message: str = ""


@dataclass
class IqmMappingOutcome:
    """Shared by iqm_preview_execution / iqm_submit_execution."""

    ok: bool
    error_message: str = ""
    backend: str = ""
    dut_label: str = ""
    healthy: bool | None = None
    placement: PlacementResult | None = None
    names: list[str] = field(default_factory=list)  # index -> QB name
    two_qubit_gate_count: int = 0
    estimated_fidelity: float = 0.0
    svg: str = ""
    circuit: Any = None
    fetched_at: float = 0.0
    age_seconds: float = 0.0
    stale: bool = False

    def physical_names(self) -> dict[int, str]:
        assert self.placement is not None
        return {lq: self.names[pq] for lq, pq in self.placement.assignment.items()}


def compute_iqm_mapping(  # noqa: PLR0911, PLR0912 -- distinct validation/error paths
    *,
    circuit_qasm: str | None,
    circuit_preset: str | None,
    num_qubits: int | None,
    qaoa_layers: int | None,
    objective: str,
    include_svg: bool = True,
) -> IqmMappingOutcome:
    if objective not in _VALID_OBJECTIVES:
        return IqmMappingOutcome(
            ok=False, error_message=f"objective must be one of {_VALID_OBJECTIVES}"
        )
    try:
        spec = parse_circuit_input(
            circuit_qasm=circuit_qasm,
            circuit_preset=circuit_preset,
            num_qubits=num_qubits,
            qaoa_layers=qaoa_layers,
        )
    except ImportError as exc:
        return IqmMappingOutcome(
            ok=False,
            error_message=f"qiskit is not installed. {IBM_EXTRA_INSTALL_HINT} ({exc})",
        )
    except (ValueError, OverflowError, ArithmeticError) as exc:
        return IqmMappingOutcome(ok=False, error_message=str(exc))

    try:
        cache = get_snapshot()
        snap = cache.payload
        hw = build_hardware_graph(snap)
    except IQMCredentialsError as exc:
        return IqmMappingOutcome(ok=False, error_message=str(exc))
    except Exception as exc:
        return IqmMappingOutcome(ok=False, error_message=f"Failed to load IQM device: {exc}")

    n_hw = hw.graph.number_of_nodes()
    if spec.num_logical_qubits > n_hw:
        return IqmMappingOutcome(
            ok=False,
            error_message=(
                f"Circuit needs {spec.num_logical_qubits} qubits but the IQM device has "
                f"only {n_hw}."
            ),
        )

    try:
        interaction = build_interaction_graph(spec.circuit)
        placement = place(
            hw.graph, interaction.graph, HardwareQualityModel(hw.graph, objective=objective)
        )
        two_q = sum(
            1
            for instr in spec.circuit.data
            if len(instr.qubits) == 2  # noqa: PLR2004
            and instr.operation.name not in _NON_INTERACTION_GATE_NAMES
        )
        # Same estimate as ibm_map_circuit: product of the CZ fidelities on the used
        # couplers and the readout fidelities of the mapped qubits.
        fidelity = 1.0
        for u, v in interaction.graph.edges:
            pu, pv = placement.assignment.get(u), placement.assignment.get(v)
            if pu is not None and pv is not None and hw.graph.has_edge(pu, pv):
                f = hw.graph.edges[pu, pv].get("gate_fidelity")
                if f is not None:
                    fidelity *= f
        for pq in placement.assignment.values():
            f = hw.graph.nodes[pq].get("readout_fidelity")
            if f is not None:
                fidelity *= f
        names = [hw.graph.nodes[i]["name"] for i in range(n_hw)]
        svg = ""
        if include_svg:
            svg = render_coupling_diagram(
                named_graph(hw), {names[pq] for pq in placement.assignment.values()}
            )
    except Exception as exc:
        return IqmMappingOutcome(ok=False, error_message=f"Placement failed: {exc}")

    return IqmMappingOutcome(
        ok=True,
        backend=snap["quantum_computer"],
        dut_label=snap["dut_label"],
        healthy=snap.get("healthy"),
        placement=placement,
        names=names,
        two_qubit_gate_count=two_q,
        estimated_fidelity=fidelity,
        svg=svg,
        circuit=spec.circuit,
        fetched_at=cache.fetched_at,
        age_seconds=cache.age_seconds,
        stale=cache.stale,
    )


def iqm_map_circuit(inp: IqmMapCircuitInput) -> IqmMapCircuitOutput:
    out = compute_iqm_mapping(
        circuit_qasm=inp.circuit_qasm,
        circuit_preset=inp.circuit_preset,
        num_qubits=inp.num_qubits,
        qaoa_layers=inp.qaoa_layers,
        objective=inp.objective,
    )
    if not out.ok or out.placement is None:
        return IqmMapCircuitOutput(ok=False, error_message=out.error_message)
    l2p = out.physical_names()
    return IqmMapCircuitOutput(
        ok=True,
        backend=out.backend,
        dut_label=out.dut_label,
        physical_qubits=sorted(l2p.values(), key=lambda n: int(n[2:])),
        logical_to_physical={str(k): v for k, v in sorted(l2p.items())},
        strategy_used=out.placement.strategy_used,
        two_qubit_gate_count=out.two_qubit_gate_count,
        estimated_fidelity=out.estimated_fidelity,
        coupling_map_svg=out.svg,
        fetched_at=dt.datetime.fromtimestamp(out.fetched_at, tz=dt.UTC).isoformat(),
        age_seconds=int(out.age_seconds),
        stale=out.stale,
    )


__all__ = [
    "IqmMapCircuitInput",
    "IqmMapCircuitOutput",
    "IqmMappingOutcome",
    "compute_iqm_mapping",
    "iqm_map_circuit",
]
