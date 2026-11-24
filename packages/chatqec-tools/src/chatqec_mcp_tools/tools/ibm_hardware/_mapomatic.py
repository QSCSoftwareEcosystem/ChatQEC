"""mapomatic layout selection, shared by the IBM and IQM mapping tools.

mapomatic (qiskit-community/mapomatic) picks a layout *after* routing: transpile the circuit
once, deflate it to its active qubits, enumerate every subgraph-isomorphic placement of that
routed circuit on the device (VF2), score each one, keep the best. Unlike the default
placement engine (_placement.place), the score sees every gate of the routed circuit,
including the CZs that SWAP insertion added.

Only mapomatic's ``matching_layouts`` is used. Its ``deflate_circuit`` / ``active_bits``
unpack CircuitInstruction as a tuple, which qiskit deprecated in 1.2 and removes in 3.0
(this repo's pytest turns that DeprecationWarning into an error), so ``deflate`` below
does the same reduction with the named attributes. Its
``evaluate_layouts`` / ``best_overall_layout`` / ``default_cost`` call
``backend.configuration()`` and ``backend.properties()``, which BackendV2 objects no longer
have (AttributeError on qiskit 2.x, checked with mapomatic 0.14.0 and qiskit 2.6.1), and
``default_cost`` counts only sx/x as single-qubit gates, so IQM's native ``r`` gates would
cost nothing. ``fidelity_cost`` below has mapomatic's cost-function shape
``(circ, layouts, backend) -> [(layout, error), ...]`` but reads errors from this package's
hardware graph, so it works for both vendors.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import networkx as nx

# rz is virtual on both vendors' hardware (a frame change), so it costs nothing.
_FREE_GATES = frozenset({"rz", "id", "delay", "barrier", "global_phase"})
_MEASURE_GATES = frozenset({"measure", "reset"})
_SEED = 1234  # fixed transpiler seed: the same question gives the same layout
_VF2_CALL_LIMIT = 1_000_000
_MAX_SCORED_LAYOUTS = 50_000


@dataclass
class MapomaticResult:
    assignment: dict[int, int]  # original logical qubit -> physical qubit
    layout: list[int]  # routed (deflated) qubit i -> physical qubit
    routed_circuit: Any  # the deflated, routed circuit, ready to run with ``layout``
    two_qubit_gate_count: int
    estimated_fidelity: float
    num_candidate_layouts: int
    strategy_used: str = "mapomatic"


def _gate_profile(circ: Any) -> tuple[Counter[int], Counter[tuple[int, int]], Counter[int]]:
    """Per-qubit 1q gate counts, per-pair 2q gate counts, per-qubit measure counts."""
    one_q: Counter[int] = Counter()
    two_q: Counter[tuple[int, int]] = Counter()
    meas: Counter[int] = Counter()
    for instr in circ.data:
        name = instr.operation.name
        idx = [circ.find_bit(q).index for q in instr.qubits]
        if name in _FREE_GATES:
            continue
        if name in _MEASURE_GATES:
            meas[idx[0]] += 1
        elif len(idx) == 1:
            one_q[idx[0]] += 1
        elif len(idx) == 2:  # noqa: PLR2004
            two_q[(idx[0], idx[1])] += 1
    return one_q, two_q, meas


def _log_fid(error: Any) -> float:
    if error is None:
        return 0.0  # unknown error: no penalty rather than a guess
    return math.log(max(1e-12, 1.0 - float(error)))


def fidelity_cost(
    circ: Any, layouts: list[list[int]], backend: nx.Graph
) -> list[tuple[list[int], float]]:
    """mapomatic-style cost function: error = 1 - product of (1 - error) over every
    physical 1q gate, 2q gate and measurement of ``circ`` placed by each layout.
    ``backend`` is the hardware graph (node attrs one_qubit_error, readout_error;
    edge attr gate_error). A layout using a missing coupler gets error 1."""
    one_q, two_q, meas = _gate_profile(circ)
    nodes, edges = backend.nodes, backend.edges
    out = []
    for layout in layouts:
        log_f = 0.0
        for q, n in one_q.items():
            log_f += n * _log_fid(nodes[layout[q]].get("one_qubit_error"))
        for q, n in meas.items():
            log_f += n * _log_fid(nodes[layout[q]].get("readout_error"))
        for (a, b), n in two_q.items():
            pa, pb = layout[a], layout[b]
            if not backend.has_edge(pa, pb):
                log_f = float("-inf")
                break
            log_f += n * _log_fid(edges[pa, pb].get("gate_error"))
        out.append((layout, 1.0 - math.exp(log_f)))
    return out


def deflate(circ: Any) -> tuple[Any, list[int]]:
    """mapomatic's deflate_circuit without the deprecated tuple access: keep only the
    qubits some non-barrier/delay instruction touches, in physical-index order. Returns
    the reduced circuit and the sorted original indices of its qubits."""
    from qiskit import QuantumCircuit  # noqa: PLC0415
    from qiskit.circuit import Barrier  # noqa: PLC0415

    idle = {"barrier", "delay"}
    active_q = sorted(
        {
            circ.find_bit(q).index
            for i in circ.data
            if i.operation.name not in idle
            for q in i.qubits
        }
    )
    active_c = sorted(
        {
            circ.find_bit(c).index
            for i in circ.data
            if i.operation.name not in idle
            for c in i.clbits
        }
    )
    q_map = {p: k for k, p in enumerate(active_q)}
    c_map = {p: k for k, p in enumerate(active_c)}
    small = QuantumCircuit(len(active_q), len(active_c))
    for instr in circ.data:
        qargs = [
            q_map[circ.find_bit(q).index] for q in instr.qubits if circ.find_bit(q).index in q_map
        ]
        if not qargs:
            continue
        if instr.operation.name == "barrier":
            small.append(Barrier(len(qargs)), qargs)
            continue
        cargs = [c_map[circ.find_bit(c).index] for c in instr.clbits]
        small.append(instr.operation, qargs, cargs)
    small.global_phase = circ.global_phase
    return small, active_q


def mapomatic_place(
    circuit: Any, hw_graph: nx.Graph, *, backend: Any = None, target: Any = None
) -> MapomaticResult:
    """Transpile ``circuit`` for the device (``backend`` or ``target``), then pick the
    lowest-error placement of the routed circuit with mapomatic. Raises ValueError when no
    layout fits, ImportError when mapomatic/qiskit are missing."""
    import mapomatic as mm  # noqa: PLC0415
    from qiskit import transpile  # noqa: PLC0415
    from qiskit.transpiler import CouplingMap  # noqa: PLC0415

    kwargs: dict[str, Any] = {"backend": backend} if backend is not None else {"target": target}
    routed = transpile(circuit, optimization_level=3, seed_transpiler=_SEED, **kwargs)
    small, active = deflate(routed)
    if small.num_qubits == 0:
        raise ValueError("The circuit has no active qubits after transpilation.")

    edges = list(hw_graph.edges)
    cmap = CouplingMap([*edges, *((b, a) for a, b in edges)])
    layouts = mm.matching_layouts(small, cmap, strict_direction=False, call_limit=_VF2_CALL_LIMIT)
    if not layouts:
        raise ValueError("mapomatic found no layout for the routed circuit on this device.")
    scored = fidelity_cost(small, layouts[:_MAX_SCORED_LAYOUTS], hw_graph)
    best_layout, best_error = min(scored, key=lambda item: (item[1], item[0]))

    # original logical qubit -> its initial physical qubit in ``routed`` -> its index in
    # ``small`` (deflate keeps active qubits in physical-index order) -> chosen physical qubit
    rank = {p: i for i, p in enumerate(active)}
    initial = routed.layout.initial_virtual_layout(filter_ancillas=True)
    assignment = {}
    for virtual in circuit.qubits:
        p0 = initial[virtual]
        if p0 in rank:
            assignment[circuit.find_bit(virtual).index] = best_layout[rank[p0]]

    two_q = sum(1 for i in small.data if len(i.qubits) == 2 and i.operation.name not in _FREE_GATES)  # noqa: PLR2004
    return MapomaticResult(
        assignment=assignment,
        layout=list(best_layout),
        routed_circuit=small,
        two_qubit_gate_count=two_q,
        estimated_fidelity=1.0 - best_error,
        num_candidate_layouts=len(layouts),
    )


def iqm_target(hw_graph: nx.Graph) -> Any:
    """A qiskit Target for an IQM device graph: r on every qubit, cz on every coupler
    (both directions), measure; errors from the graph."""
    from qiskit.circuit import Measure, Parameter  # noqa: PLC0415
    from qiskit.circuit.library import CZGate, RGate  # noqa: PLC0415
    from qiskit.transpiler import InstructionProperties, Target  # noqa: PLC0415

    nodes = sorted(hw_graph.nodes)
    target = Target(num_qubits=len(nodes))
    target.add_instruction(
        RGate(Parameter("theta"), Parameter("phi")),
        {
            (q,): InstructionProperties(error=hw_graph.nodes[q].get("one_qubit_error"))
            for q in nodes
        },
    )
    cz = {}
    for a, b, d in hw_graph.edges(data=True):
        props = InstructionProperties(error=d.get("gate_error"), duration=d.get("gate_time"))
        cz[(a, b)] = props
        cz[(b, a)] = props
    target.add_instruction(CZGate(), cz)
    target.add_instruction(
        Measure(),
        {(q,): InstructionProperties(error=hw_graph.nodes[q].get("readout_error")) for q in nodes},
    )
    return target


__all__ = ["MapomaticResult", "fidelity_cost", "iqm_target", "mapomatic_place"]
