"""Uniform circuit-level noise for noiseless Stim circuits (e.g. LightStim output).

Same noise model as ``stim.Circuit.generated(after_clifford_depolarization=p,
before_measure_flip_probability=p, after_reset_flip_probability=p)``:
DEPOLARIZE1/2(p) after each 1/2-qubit Clifford gate, a flip with probability p
before each measurement and after each reset (Z_ERROR for X-basis operations,
X_ERROR otherwise). REPEAT blocks are kept, so detector structure is unchanged.
Annotations, existing noise channels, and Pauli-product ops (MPP, SPP) pass
through untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import stim

_PASS_THROUGH = {"MPP", "SPP", "SPP_DAG"}


def _flip(name: str) -> str:
    """Error channel that flips a measurement/reset in the given gate's basis."""
    # RX/MX/MRX live in the X basis, so a Z flips them; X flips Z- and Y-basis ops.
    return "Z_ERROR" if name.endswith("X") else "X_ERROR"


def _is_noise_channel(name: str, data: stim.GateData) -> bool:
    # M/MR also report is_noisy_gate (they take a flip-probability argument),
    # so exclude measurements other than the heralded noise channels.
    measures = data.produces_measurements and not name.startswith("HERALDED")
    return data.is_noisy_gate and not data.is_unitary and not data.is_reset and not measures


def add_uniform_noise(circuit: stim.Circuit, p: float) -> stim.Circuit:
    import stim  # noqa: PLC0415

    out = stim.Circuit()
    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            out.append(
                stim.CircuitRepeatBlock(inst.repeat_count, add_uniform_noise(inst.body_copy(), p))
            )
            continue
        data = stim.gate_data(inst.name)
        targets = inst.targets_copy()
        qubits = [t.value for t in targets if t.is_qubit_target]
        if inst.name in _PASS_THROUGH or not qubits or _is_noise_channel(inst.name, data):
            out.append(inst)
        elif data.produces_measurements:
            out.append(_flip(inst.name), qubits, p)
            out.append(inst)
            if data.is_reset:  # MR / MRX / MRY: also flip after the reset half
                out.append(_flip(inst.name), qubits, p)
        elif data.is_reset:
            out.append(inst)
            out.append(_flip(inst.name), qubits, p)
        elif data.is_unitary:
            out.append(inst)
            if data.is_two_qubit_gate and len(qubits) == len(targets):
                out.append("DEPOLARIZE2", qubits, p)
            else:  # 1-qubit gates, or classically controlled 2-qubit gates
                out.append("DEPOLARIZE1", qubits, p)
        else:
            out.append(inst)
    return out
