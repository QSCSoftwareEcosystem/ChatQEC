"""ibm_ft_compile: Qiskit >= 2.5 fault-tolerant compilation (Clifford+T via gridsynth, or PBC)."""

import builtins

import pytest

from chatqec_mcp_tools.tools.ibm_hardware.ft_compile import IbmFtCompileInput, ibm_ft_compile

pytest.importorskip("qiskit")
try:
    from qiskit.transpiler import generate_preset_clifford_t_pass_manager  # noqa: F401
except ImportError:
    pytest.skip("qiskit >= 2.5 (ft extra) not installed", allow_module_level=True)

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
CLIFFORD_T_GATES = {"h", "s", "sdg", "t", "tdg", "x", "y", "z", "sx", "sxdg", "cx", "cz", "swap"}


def _qasm(body: str, qubits: int = 1) -> str:
    return f"{HEADER}qreg q[{qubits}];\n{body}\n"


def _compile(body: str, qubits: int = 1, **kwargs):
    return ibm_ft_compile(IbmFtCompileInput(circuit_qasm=_qasm(body, qubits), **kwargs))


def test_rz_pi_over_4_is_exactly_one_t_gate():
    out = _compile("rz(pi/4) q[0];")
    assert out.ok, out.error_message
    assert out.mode == "clifford_t" and out.t_count == 1


def test_clifford_only_circuit_has_zero_t_gates():
    out = _compile("h q[0];\ncx q[0],q[1];\ns q[1];", qubits=2)
    assert out.ok and out.t_count == 0 and out.clifford_count == 3


def test_output_is_clifford_t_only_and_parses_back():
    from qiskit import QuantumCircuit

    out = _compile("rz(0.3) q[0];", gridsynth_precision=3)
    assert out.ok, out.error_message
    names = set(QuantumCircuit.from_qasm_str(out.circuit_qasm).count_ops())
    assert names <= CLIFFORD_T_GATES, names - CLIFFORD_T_GATES


def test_higher_precision_costs_more_t_gates():
    low = _compile("rz(0.3) q[0];", gridsynth_precision=2)
    high = _compile("rz(0.3) q[0];", gridsynth_precision=6)
    assert low.ok and high.ok
    assert high.t_count > low.t_count


def test_pbc_mode_reports_rotations_not_t_gates():
    out = _compile("h q[0];\nt q[0];\ncx q[0],q[1];\nrz(0.7) q[1];", qubits=2, mode="pbc")
    assert out.ok, out.error_message
    assert out.mode == "pbc" and out.num_rotations and out.num_rotations >= 1
    assert out.t_count is None


def test_invalid_qasm_reports_a_parse_error():
    out = ibm_ft_compile(IbmFtCompileInput(circuit_qasm="not qasm"))
    assert out.ok is False and "parse" in out.error_message.lower()


def test_empty_circuit_is_rejected():
    out = ibm_ft_compile(IbmFtCompileInput(circuit_qasm=_qasm("", qubits=0)))
    assert out.ok is False and "0 qubits" in out.error_message


def test_oversized_circuit_is_rejected_before_parsing():
    out = ibm_ft_compile(IbmFtCompileInput(circuit_qasm="x" * (300 * 1024)))
    assert out.ok is False and "too large" in out.error_message


def test_too_many_qubits_is_rejected():
    out = ibm_ft_compile(IbmFtCompileInput(circuit_qasm=_qasm("h q[0];", qubits=200)))
    assert out.ok is False and "qubits" in out.error_message


def test_old_qiskit_points_at_the_ft_extra(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "qiskit.transpiler" and "generate_preset_clifford_t_pass_manager" in (
            fromlist or ()
        ):
            raise ImportError("old qiskit")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    out = _compile("rz(pi/4) q[0];")
    assert out.ok is False and "[ft]" in out.error_message and "2.5" in out.error_message


def test_large_compiled_circuit_omits_the_qasm_but_keeps_the_counts():
    """Thousands of T gates would flood the LLM context; return the numbers, not the text."""
    body = "".join(
        f"rz({0.1 * (i + 1):.3f}) q[{i % 4}];\ncx q[{i % 4}],q[{(i + 1) % 4}];\n" for i in range(30)
    )
    out = _compile(body, qubits=4, gridsynth_precision=10)
    assert out.ok and out.t_count > 1000
    assert out.circuit_qasm == "" and "omitted" in out.notes
