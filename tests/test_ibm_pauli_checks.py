"""ibm_add_pauli_checks: spacetime Pauli checks (qiskit-paulice) with overhead reporting."""

import builtins

import pytest

from chatqec_mcp_tools.tools.ibm_hardware.pauli_checks import (
    IbmAddPauliChecksInput,
    ibm_add_pauli_checks,
)

pytest.importorskip("qiskit_paulice")
pytest.importorskip("qiskit_aer")

GHZ3_QASM = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0],q[1];
cx q[1],q[2];
measure q -> c;
"""
GHZ3_STIM = "H 0\nCX 0 1\nCX 1 2\nM 0 1 2"


def _run(**kwargs):
    return ibm_add_pauli_checks(IbmAddPauliChecksInput(circuit=GHZ3_QASM, seed=1, **kwargs))


def test_adds_checks_to_ghz3_and_reports_overhead():
    from qiskit import QuantumCircuit

    out = _run()
    assert out.ok, out.error_message
    o = out.overhead
    assert o.qubits_before == 3 and o.qubits_after > o.qubits_before
    assert o.gates_after > o.gates_before and o.depth_after >= o.depth_before
    assert QuantumCircuit.from_qasm_str(out.circuit_qasm).num_qubits == o.qubits_after


def test_stim_input_is_converted():
    out = ibm_add_pauli_checks(
        IbmAddPauliChecksInput(circuit=GHZ3_STIM, circuit_format="stim", seed=1)
    )
    assert out.ok, out.error_message
    assert out.overhead.qubits_before == 3


def test_num_checks_zero_leaves_the_circuit_unchanged_in_size():
    out = _run(num_checks=0)
    assert out.ok and out.overhead.qubits_after == out.overhead.qubits_before


def test_num_checks_one_adds_exactly_one_check_qubit():
    out = _run(num_checks=1)
    assert out.ok and out.overhead.qubits_after == out.overhead.qubits_before + 1


def test_discard_rate_is_none_without_noise():
    assert _run().postselection_discard_rate is None


def test_discard_rate_is_a_probability_with_noise():
    out = _run(noise_p=0.02, shots=1000)
    assert out.ok and out.postselection_discard_rate is not None
    assert 0.0 <= out.postselection_discard_rate <= 1.0


def test_discard_rate_grows_with_noise():
    low = _run(noise_p=0.001, shots=2000).postselection_discard_rate
    high = _run(noise_p=0.05, shots=2000).postselection_discard_rate
    assert low is not None and high is not None
    assert low < 0.05  # near-noiseless: a wrong bit order would flag most shots
    assert high > low


def test_non_clifford_circuit_is_rejected_naming_the_cause():
    qasm = GHZ3_QASM.replace("h q[0];", "t q[0];")
    out = ibm_add_pauli_checks(IbmAddPauliChecksInput(circuit=qasm))
    assert out.ok is False and "Clifford" in out.error_message


def test_circuit_without_terminal_measurement_is_rejected():
    qasm = GHZ3_QASM.replace("measure q -> c;\n", "")
    out = ibm_add_pauli_checks(IbmAddPauliChecksInput(circuit=qasm))
    assert out.ok is False and "measure" in out.error_message.lower()


def test_invalid_qasm_reports_a_parse_error():
    out = ibm_add_pauli_checks(IbmAddPauliChecksInput(circuit="this is not qasm"))
    assert out.ok is False and "parse" in out.error_message.lower()


def test_stim_circuit_with_noise_instructions_is_rejected():
    out = ibm_add_pauli_checks(
        IbmAddPauliChecksInput(circuit="H 0\nX_ERROR(0.1) 0\nM 0", circuit_format="stim")
    )
    assert out.ok is False and "noise" in out.error_message.lower()


def test_oversized_circuit_is_rejected_before_parsing():
    out = ibm_add_pauli_checks(IbmAddPauliChecksInput(circuit="x" * (300 * 1024)))
    assert out.ok is False and "too large" in out.error_message


def test_too_many_qubits_is_rejected():
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[40];\ncreg c[40];\nmeasure q -> c;\n'
    out = ibm_add_pauli_checks(IbmAddPauliChecksInput(circuit=qasm))
    assert out.ok is False and "qubits" in out.error_message


def test_missing_extra_reports_the_paulice_hint(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.split(".")[0] in ("qiskit_paulice", "qiskit_aer"):
            raise ImportError("missing")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    out = _run()
    assert out.ok is False and "[paulice]" in out.error_message


def test_large_checked_circuit_omits_the_qasm_but_keeps_the_overhead():
    """A big circuit's QASM would flood the LLM context; return the numbers, not the text."""
    body = "h q[0];\n" + "cx q[0],q[1];\ncx q[1],q[2];\n" * 1100 + "measure q -> c;\n"
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c[3];\n' + body
    out = ibm_add_pauli_checks(IbmAddPauliChecksInput(circuit=qasm, num_checks=1, seed=1))
    assert out.ok, out.error_message
    assert out.overhead.gates_after > 2200
    assert out.circuit_qasm == "" and "omitted" in out.notes
