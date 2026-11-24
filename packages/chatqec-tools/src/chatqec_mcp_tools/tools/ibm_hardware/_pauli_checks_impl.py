"""Helpers for ibm_add_pauli_checks (all heavy imports are lazy: ``[paulice]`` extra).

qiskit-paulice 0.2.1 API, as probed:
  ``add_pauli_checks(circuit, target_qubits, noise_model, cost="gamma", ..., seed=None)``
  takes a CLIFFORD ``QuantumCircuit`` that ends in measurements and returns a list of
  ``CheckedCircuit``; entry ``k`` carries ``k`` checks (entry 0 has none). Each has
  ``.circuit`` (a QuantumCircuit with extra ``checks_q`` / ``checks_c`` registers),
  ``.check_qubits``, ``.cost`` (the gamma sampling overhead, lower is better) and
  ``.get_postselection_method()``, a function mapping one shot's bitstring to a syndrome
  vector that is all zeros iff no check fired. Noise comes from
  ``NoiseModel(gate_noise=UniformGateNoise(p), readout_noise=..., idling_noise=...)``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from chatqec_mcp_tools.tools.ibm_hardware._circuits import parse_circuit_input

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

_NON_GATES = frozenset({"measure", "barrier"})
_STIM_1Q = {"H": "h", "S": "s", "S_DAG": "sdg", "X": "x", "Y": "y", "Z": "z", "I": "id"}
_STIM_2Q = {"CX": "cx", "CNOT": "cx", "ZCX": "cx", "CZ": "cz", "SWAP": "swap"}
_STIM_MEASURE = frozenset({"M", "MZ"})
_STIM_NOISE_HINTS = ("ERROR", "DEPOLARIZE")
_DEFAULT_PICK_NOISE = 0.001
_IDLE_FRACTION = 0.1  # idling noise assumed one tenth of the gate noise when picking checks


def _stim_to_qiskit(text: str) -> QuantumCircuit:
    import stim  # noqa: PLC0415
    from qiskit import QuantumCircuit  # noqa: PLC0415

    try:
        circuit = stim.Circuit(text).flattened()
    except ValueError as exc:
        raise ValueError(f"Could not parse the Stim circuit: {exc}") from exc
    qc = QuantumCircuit(circuit.num_qubits, circuit.num_measurements)
    clbit = 0
    for inst in circuit:
        name = inst.name
        targets = [t.value for t in inst.targets_copy()]
        if any(h in name for h in _STIM_NOISE_HINTS):
            raise ValueError(
                f"the Stim circuit has a noise instruction ({name}); pass it noiseless and "
                "set noise_p to model noise"
            )
        if name in _STIM_1Q:
            for q in targets:
                getattr(qc, _STIM_1Q[name])(q)
        elif name in _STIM_2Q:
            for a, b in zip(targets[::2], targets[1::2], strict=True):
                getattr(qc, _STIM_2Q[name])(a, b)
        elif name in _STIM_MEASURE:
            for q in targets:
                qc.measure(q, clbit)
                clbit += 1
        elif name not in ("TICK", "QUBIT_COORDS"):
            raise ValueError(f"unsupported Stim instruction for Pauli checks: {name}")
    return qc


def load_circuit(text: str, circuit_format: str) -> QuantumCircuit:
    """Parse the input and check Paulice's preconditions.

    Raises:
        ValueError: unparsable, not Clifford, not terminated by a measurement, or too wide.
    """
    from qiskit.quantum_info import Clifford  # noqa: PLC0415

    if circuit_format == "stim":
        circuit = _stim_to_qiskit(text)
    else:
        circuit = parse_circuit_input(
            circuit_qasm=text, circuit_preset=None, num_qubits=None, qaoa_layers=None
        ).circuit
    if not any(inst.operation.name == "measure" for inst in circuit.data):
        raise ValueError("the circuit must end with measurements; add measure instructions")
    try:
        Clifford(circuit.remove_final_measurements(inplace=False))
    except Exception as exc:
        raise ValueError(
            f"spacetime Pauli checks need a Clifford circuit (H, S, CX, CZ, Paulis); got: {exc}"
        ) from exc
    return circuit


def count_gates(circuit: QuantumCircuit) -> int:
    return sum(n for name, n in circuit.count_ops().items() if name not in _NON_GATES)


def pick_checks(
    circuit: QuantumCircuit, num_checks: int | None, noise_p: float | None, seed: int | None
) -> tuple[Any, str]:
    """Run Paulice and return ``(CheckedCircuit, note)``.

    With ``num_checks`` None the lowest-cost candidate wins; otherwise the candidate with that
    many checks (clamped to what the search found).
    """
    import qiskit_paulice as qp  # noqa: PLC0415
    from qiskit_paulice.noise_models import UniformGateNoise  # noqa: PLC0415

    p = noise_p or _DEFAULT_PICK_NOISE
    noise = qp.NoiseModel(
        gate_noise=UniformGateNoise(p), readout_noise=p, idling_noise=p * _IDLE_FRACTION
    )
    # Paulice warns when it cannot express the result in the input's gate set (e.g. a circuit
    # of only H and CX); that is expected, so surface it as a note instead of raising.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        candidates = qp.add_pauli_checks(
            circuit, target_qubits=list(range(circuit.num_qubits)), noise_model=noise, seed=seed
        )
    notes = sorted({str(w.message) for w in caught if issubclass(w.category, UserWarning)})
    if num_checks is None:
        chosen = min(candidates, key=lambda c: c.cost)
    else:
        index = min(num_checks, len(candidates) - 1)
        chosen = candidates[index]
        if index != num_checks:
            notes.append(f"only {index} valid check(s) found; asked for {num_checks}")
    return chosen, " ".join(notes)


def discard_rate(checked: Any, noise_p: float, shots: int, seed: int | None) -> float:
    """Fraction of noisy shots in which at least one check fires (those shots are discarded)."""
    from qiskit import transpile  # noqa: PLC0415
    from qiskit_aer import AerSimulator  # noqa: PLC0415
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error  # noqa: PLC0415

    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(
        depolarizing_error(noise_p, 1), ["h", "s", "sdg", "sx", "sxdg", "x", "y", "z", "id"]
    )
    noise.add_all_qubit_quantum_error(depolarizing_error(noise_p, 2), ["cx", "cz", "swap"])
    noise.add_all_qubit_readout_error(
        ReadoutError([[1 - noise_p, noise_p], [noise_p, 1 - noise_p]])
    )
    sim = AerSimulator(method="stabilizer", noise_model=noise, seed_simulator=seed)
    compiled = transpile(checked.circuit, sim, optimization_level=0)
    memory = sim.run(compiled, shots=shots, memory=True).result().get_memory()
    syndrome = checked.get_postselection_method()
    flagged = sum(1 for shot in memory if syndrome(shot.replace(" ", "")).any())
    return flagged / shots


def to_qasm(circuit: QuantumCircuit) -> str:
    from qiskit import qasm2, qasm3  # noqa: PLC0415

    try:
        return str(qasm2.dumps(circuit))
    except Exception:
        return str(qasm3.dumps(circuit))
