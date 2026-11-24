from __future__ import annotations

import pytest


def test_parse_circuit_input_from_qasm():
    from chatqec_mcp_tools.tools.ibm_hardware._circuits import parse_circuit_input

    qasm = (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
        "h q[0];\ncx q[0],q[1];\nmeasure q -> c;\n"
    )
    spec = parse_circuit_input(
        circuit_qasm=qasm, circuit_preset=None, num_qubits=None, qaoa_layers=None
    )

    assert spec.num_logical_qubits == 2


def test_parse_circuit_input_rejects_both_qasm_and_preset():
    from chatqec_mcp_tools.tools.ibm_hardware._circuits import parse_circuit_input

    with pytest.raises(ValueError, match="exactly one"):
        parse_circuit_input(
            circuit_qasm="OPENQASM 2.0;", circuit_preset="bell", num_qubits=None, qaoa_layers=None
        )


def test_parse_circuit_input_rejects_neither():
    from chatqec_mcp_tools.tools.ibm_hardware._circuits import parse_circuit_input

    with pytest.raises(ValueError, match="exactly one"):
        parse_circuit_input(
            circuit_qasm=None, circuit_preset=None, num_qubits=None, qaoa_layers=None
        )


def test_bell_preset_is_two_qubits():
    from chatqec_mcp_tools.tools.ibm_hardware._circuits import parse_circuit_input

    spec = parse_circuit_input(
        circuit_qasm=None, circuit_preset="bell", num_qubits=None, qaoa_layers=None
    )

    assert spec.num_logical_qubits == 2
    assert spec.circuit.count_ops().get("cx", 0) == 1


def test_ghz_preset_respects_num_qubits():
    from chatqec_mcp_tools.tools.ibm_hardware._circuits import parse_circuit_input

    spec = parse_circuit_input(
        circuit_qasm=None, circuit_preset="ghz", num_qubits=5, qaoa_layers=None
    )

    assert spec.num_logical_qubits == 5
    assert spec.circuit.count_ops().get("cx", 0) == 4


def test_qaoa_preset_layers_multiply_two_qubit_gate_count():
    from chatqec_mcp_tools.tools.ibm_hardware._circuits import parse_circuit_input

    one_layer = parse_circuit_input(
        circuit_qasm=None, circuit_preset="qaoa", num_qubits=4, qaoa_layers=1
    )
    two_layers = parse_circuit_input(
        circuit_qasm=None, circuit_preset="qaoa", num_qubits=4, qaoa_layers=2
    )

    assert two_layers.circuit.count_ops().get("cx", 0) == 2 * one_layer.circuit.count_ops().get(
        "cx", 0
    )


def test_build_interaction_graph_weights_repeated_pairs():
    from chatqec_mcp_tools.tools.ibm_hardware._circuits import (
        build_interaction_graph,
        parse_circuit_input,
    )

    spec = parse_circuit_input(
        circuit_qasm=None, circuit_preset="ghz", num_qubits=3, qaoa_layers=None
    )
    interaction = build_interaction_graph(spec.circuit)

    assert set(interaction.graph.nodes) == {0, 1, 2}
    assert interaction.graph.has_edge(0, 1)
    assert interaction.graph.has_edge(0, 2)


def test_qaoa_layers_zero_produces_empty_circuit():
    from chatqec_mcp_tools.tools.ibm_hardware._circuits import parse_circuit_input

    spec = parse_circuit_input(
        circuit_qasm=None, circuit_preset="qaoa", num_qubits=4, qaoa_layers=0
    )

    # 0 layers should produce a circuit with no QAOA construction gates (h, cx, rz, rx)
    ops = spec.circuit.count_ops()
    assert ops.get("h", 0) == 0
    assert ops.get("cx", 0) == 0
    assert ops.get("rz", 0) == 0
    assert ops.get("rx", 0) == 0


# --------------------------------------------------------------------------
# 0-qubit circuits must never reach the credit-spending submission path.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "qasm",
    [
        "",
        "   \n\t ",
        'OPENQASM 2.0;\ninclude "qelib1.inc";\n',
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[0];\n',
    ],
    ids=["empty", "whitespace", "no-qreg", "zero-width-qreg"],
)
def test_parse_circuit_input_rejects_zero_qubit_qasm(qasm):
    """QuantumCircuit.from_qasm_str does NOT raise on any of these -- it returns
    a perfectly valid 0-qubit circuit. Since Task 13 wired spec.circuit straight
    into ibm_submit_execution's sampler.run, such a circuit reached real
    hardware and spent credits on a job that runs nothing while reporting
    ok=True with a job_id. parse_circuit_input is the single chokepoint all
    three circuit tools share, so the rejection belongs here."""
    from chatqec_mcp_tools.tools.ibm_hardware._circuits import parse_circuit_input

    with pytest.raises(ValueError, match="0 qubits"):
        parse_circuit_input(
            circuit_qasm=qasm, circuit_preset=None, num_qubits=None, qaoa_layers=None
        )


def test_parse_circuit_input_still_accepts_a_one_qubit_qasm():
    """Guard against over-rejecting: the 0-qubit check must not touch the
    smallest legitimate circuit."""
    from chatqec_mcp_tools.tools.ibm_hardware._circuits import parse_circuit_input

    spec = parse_circuit_input(
        circuit_qasm='OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nh q[0];\n',
        circuit_preset=None,
        num_qubits=None,
        qaoa_layers=None,
    )

    assert spec.num_logical_qubits == 1


def test_parse_circuit_input_preset_path_is_guarded_too():
    """The 0-qubit check covers the preset branch as well, not just QASM: the
    pydantic ge=1 bound on num_qubits guards the tools, but parse_circuit_input
    is also called directly (by compute_mapping's own tests, and by anything
    added later), so the invariant is enforced at the function itself."""
    from chatqec_mcp_tools.tools.ibm_hardware._circuits import parse_circuit_input

    with pytest.raises(ValueError, match="at least 1"):
        parse_circuit_input(circuit_qasm=None, circuit_preset="ghz", num_qubits=0, qaoa_layers=None)


def test_qaoa_preset_requires_at_least_two_qubits():
    """QAOA's ring topology uses qc.cx(i, (i+1) % num_qubits). When num_qubits=1,
    this becomes qc.cx(0, 0), a two-qubit gate with both operands the same,
    which qiskit rejects with CircuitError('duplicate bit arguments'). The
    guard must reject this before circuit construction, raising ValueError so
    compute_mapping's except clause turns it into ok=False for all three tools."""
    from chatqec_mcp_tools.tools.ibm_hardware._circuits import parse_circuit_input

    with pytest.raises(ValueError, match="at least 2 qubits"):
        parse_circuit_input(
            circuit_qasm=None, circuit_preset="qaoa", num_qubits=1, qaoa_layers=None
        )


def test_qaoa_preset_works_with_two_or_more_qubits():
    """Guard against over-rejecting: num_qubits=2 and higher should work."""
    from chatqec_mcp_tools.tools.ibm_hardware._circuits import parse_circuit_input

    spec = parse_circuit_input(
        circuit_qasm=None, circuit_preset="qaoa", num_qubits=2, qaoa_layers=1
    )

    assert spec.num_logical_qubits == 2
    # Each layer has num_qubits H gates, num_qubits*2 CX gates (cx before+after rz),
    # num_qubits RZ gates, and num_qubits RX gates. With 1 layer and 2 qubits:
    assert spec.circuit.count_ops().get("cx", 0) == 4
