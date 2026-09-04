"""Live IBM backend -> normalized property graph.

Adapted from cahgr/hardware_kg.py in the intern's
HardwareAwarenessIntegrationPipeline: same node/edge schema and the same
native-two-qubit-gate priority list, but built from a live qiskit_ibm_runtime
backend's .properties()/.coupling_map instead of a parsed JSON snapshot. The
intern's separate hardware_layout.vendor_ibm (JSON reconstruction) and
hardware_layout.qubit_coordinates (per-chip coordinate table) are intentionally
not ported -- unnecessary once the source is a live SDK object, and the
coupling-map diagram (see _coupling_diagram.py) computes its own layout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import networkx as nx

# Priority order to look for on a coupler. Different chip families report
# different native gates (Heron: cz + rzz; Eagle/older chips: cx or ecr) --
# confirmed live 2026-09-03 that a single coupler can expose more than one
# (ibm_fez qubits (0,1): both cz and rzz). Picking by priority rather than
# tracking every alternative is a deliberate simplification for scoring.
_TWO_Q_GATE_PRIORITY = ("cz", "ecr", "cx", "rzz", "iswap", "swap")

# Readout error threshold above which a qubit is considered not operational.
_READOUT_ERROR_THRESHOLD = 0.5


@dataclass
class HardwareGraph:
    graph: nx.Graph
    backend_name: str

    @property
    def num_qubits(self) -> int:
        return cast(int, self.graph.number_of_nodes())


def build_hardware_graph(backend: Any) -> HardwareGraph:
    """Build the normalized hardware graph from a live backend object (anything
    with .name, .num_qubits, .coupling_map, .properties())."""
    props = backend.properties()

    graph = nx.Graph()
    graph.graph["backend_name"] = backend.name
    graph.graph["last_update_date"] = str(getattr(props, "last_update_date", None))

    for q in range(backend.num_qubits):
        qprop = props.qubit_property(q) if props is not None else {}
        t1 = _first(qprop.get("T1"))
        t2 = _first(qprop.get("T2"))
        readout_error = _first(qprop.get("readout_error"))
        graph.add_node(
            q,
            T1=t1,
            T2=t2,
            readout_error=readout_error,
            readout_fidelity=_fidelity_from_error(readout_error),
            is_operational=_is_operational(t1, t2, readout_error),
        )

    for u, v in backend.coupling_map:
        if graph.has_edge(u, v):
            continue
        metrics = (
            _pick_native_two_qubit_metrics(props, u, v) if props is not None else _empty_metrics()
        )
        graph.add_edge(u, v, **metrics)

    return HardwareGraph(graph=graph, backend_name=backend.name)


def _first(value: Any) -> float | None:
    """qiskit's qubit_property() values are (value, unit) tuples; unwrap."""
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return cast(float | None, value[0]) if value else None
    return cast(float | None, value)


def _pick_native_two_qubit_metrics(props: Any, u: int, v: int) -> dict[str, Any]:
    for gate in _TWO_Q_GATE_PRIORITY:
        for pair in ((u, v), (v, u)):
            try:
                error = props.gate_error(gate, pair)
                duration = props.gate_length(gate, pair)
            except Exception:
                continue
            return {
                "native_gate": gate,
                "gate_error": error,
                "gate_time": duration,
                "gate_fidelity": _fidelity_from_error(error),
            }
    return _empty_metrics()


def _empty_metrics() -> dict[str, Any]:
    return {"native_gate": None, "gate_error": None, "gate_time": None, "gate_fidelity": None}


def _fidelity_from_error(error: Any) -> float | None:
    if error is None:
        return None
    try:
        return max(0.0, 1.0 - float(error))
    except (TypeError, ValueError):
        return None


def _is_operational(t1: Any, t2: Any, readout_error: Any) -> bool:
    for value in (t1, t2):
        if value is None:
            return False
        try:
            if math.isnan(float(value)) or float(value) <= 0:
                return False
        except (TypeError, ValueError):
            return False
    if readout_error is not None:
        try:
            if float(readout_error) >= _READOUT_ERROR_THRESHOLD:
                return False
        except (TypeError, ValueError):
            pass
    return True


__all__ = ["HardwareGraph", "build_hardware_graph"]
