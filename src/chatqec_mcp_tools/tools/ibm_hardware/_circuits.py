"""Circuit input handling: QASM parsing, named presets, and the two-qubit
interaction graph the placement engine matches against hardware.

build_interaction_graph adapts cahgr/circuit_graph.py's
build_circuit_interaction_graph (verified 2026-09-03: it already dispatches on
a real Qiskit QuantumCircuit, so only that code path is ported here -- the
generalization to arbitrary circuits this package needs was already present in
the intern's code, not something added). Presets adapt qec/prompt_library.py's
_ghz/_qft/_qaoa (verified correct: standard GHZ/QFT/ring-MaxCut-QAOA
constructions).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import networkx as nx

_BELL_QUBITS = 2
_DEFAULT_NUM_QUBITS = 3
_PRESETS = ("bell", "ghz", "qft", "qaoa")
_TWO_QUBIT_GATES = 2


@dataclass
class CircuitSpec:
    circuit: Any  # a qiskit.circuit.QuantumCircuit
    num_logical_qubits: int


@dataclass
class InteractionGraph:
    graph: nx.Graph


def parse_circuit_input(
    *,
    circuit_qasm: str | None,
    circuit_preset: str | None,
    num_qubits: int | None,
    qaoa_layers: int | None,
) -> CircuitSpec:
    if (circuit_qasm is None) == (circuit_preset is None):
        raise ValueError("Provide exactly one of circuit_qasm or circuit_preset.")

    from qiskit import QuantumCircuit  # noqa: PLC0415

    if circuit_qasm is not None:
        try:
            circuit = QuantumCircuit.from_qasm_str(circuit_qasm)
        except Exception as exc:
            raise ValueError(f"Could not parse circuit_qasm: {exc}") from exc
        _reject_empty_circuit(circuit)
        return CircuitSpec(circuit=circuit, num_logical_qubits=circuit.num_qubits)

    if circuit_preset not in _PRESETS:
        raise ValueError(f"circuit_preset must be one of {_PRESETS}, got {circuit_preset!r}")

    n = (
        num_qubits
        if num_qubits is not None
        else (_BELL_QUBITS if circuit_preset == "bell" else _DEFAULT_NUM_QUBITS)
    )
    # Checked before construction, not after: _ghz(0)/_qft(0)/_qaoa(0) reach
    # qiskit's own CircuitError (not a ValueError), which compute_mapping's
    # except clause would not catch, so it would escape as a raw QiskitError.
    if n < 1:
        raise ValueError(f"num_qubits must be at least 1 for a circuit to run, got {n}.")

    if circuit_preset == "bell":
        if n != _BELL_QUBITS:
            raise ValueError("the bell preset is defined for exactly 2 qubits")
        circuit = _bell()
    elif circuit_preset == "ghz":
        circuit = _ghz(n)
    elif circuit_preset == "qft":
        circuit = _qft(n)
    else:
        circuit = _qaoa(n, layers=qaoa_layers if qaoa_layers is not None else 1)

    _reject_empty_circuit(circuit)
    return CircuitSpec(circuit=circuit, num_logical_qubits=n)


def _reject_empty_circuit(circuit: Any) -> None:
    """A 0-qubit circuit is not a mapping/execution request anyone means.

    QuantumCircuit.from_qasm_str("") does NOT raise -- it returns a perfectly
    valid 0-qubit circuit, as does a QASM string declaring `qreg q[0];`. Left
    unchecked that circuit flows through compute_mapping (an empty placement
    passes every downstream size check) all the way into
    ibm_submit_execution's sampler.run, spending real account credits on a job
    that runs nothing while reporting ok=True with a job_id. Raised as
    ValueError so compute_mapping's existing except clause around
    parse_circuit_input turns it into ok=False for all three callers
    (ibm_map_circuit / ibm_preview_execution / ibm_submit_execution)."""
    if circuit.num_qubits == 0:
        raise ValueError(
            "The circuit has 0 qubits, so there is nothing to map or execute. "
            "Provide a circuit_qasm that declares at least one qubit, or use a "
            "circuit_preset."
        )


def build_interaction_graph(circuit: Any) -> InteractionGraph:
    graph = nx.Graph()
    graph.add_nodes_from(range(circuit.num_qubits))
    qubit_index = {qubit: i for i, qubit in enumerate(circuit.qubits)}

    for instruction in circuit.data:
        name = instruction.operation.name
        if name in {"barrier", "measure", "reset", "delay"}:
            continue
        idxs = [qubit_index[q] for q in instruction.qubits]
        if len(idxs) == _TWO_QUBIT_GATES:
            u, v = idxs
            if graph.has_edge(u, v):
                graph.edges[u, v]["weight"] += 1
            else:
                graph.add_edge(u, v, weight=1)

    return InteractionGraph(graph=graph)


def _bell() -> Any:
    from qiskit import QuantumCircuit  # noqa: PLC0415

    qc = QuantumCircuit(2, name="bell")
    qc.h(0)
    qc.cx(0, 1)
    return qc


def _ghz(num_qubits: int) -> Any:
    from qiskit import QuantumCircuit  # noqa: PLC0415

    qc = QuantumCircuit(num_qubits, name=f"ghz{num_qubits}")
    qc.h(0)
    for i in range(1, num_qubits):
        qc.cx(0, i)
    return qc


def _qft(num_qubits: int) -> Any:
    from qiskit import QuantumCircuit  # noqa: PLC0415

    qc = QuantumCircuit(num_qubits, name=f"qft{num_qubits}")
    for j in range(num_qubits):
        qc.h(j)
        for k in range(j + 1, num_qubits):
            qc.cp(math.pi / 2 ** (k - j), k, j)
    for i in range(num_qubits // 2):
        qc.swap(i, num_qubits - i - 1)
    return qc


def _qaoa(num_qubits: int, *, layers: int) -> Any:
    from qiskit import QuantumCircuit  # noqa: PLC0415

    qc = QuantumCircuit(num_qubits, name=f"qaoa_maxcut_ring{num_qubits}_p{layers}")
    gamma, beta = math.pi / 4, math.pi / 8
    for _ in range(layers):
        for i in range(num_qubits):
            qc.h(i)
        for i in range(num_qubits):
            j = (i + 1) % num_qubits
            qc.cx(i, j)
            qc.rz(2 * gamma, j)
            qc.cx(i, j)
        for i in range(num_qubits):
            qc.rx(2 * beta, i)
    return qc


__all__ = ["CircuitSpec", "InteractionGraph", "build_interaction_graph", "parse_circuit_input"]
