from __future__ import annotations

import pytest


def test_parse_circuit_input_from_qasm():
    from chatqec_mcp_tools.tools.ibm_hardware._circuits import parse_circuit_input

    qasm = (
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg c[2];\n"
        "h q[0];\ncx q[0],q[1];\nmeasure q -> c;\n"
    )
    spec = parse_circuit_input(circuit_qasm=qasm, circuit_preset=None, num_qubits=None, qaoa_layers=None)

    assert spec.num_logical_qubits == 2


def test_parse_circuit_input_rejects_both_qasm_and_preset():
    from chatqec_mcp_tools.tools.ibm_hardware._circuits import parse_circuit_input

    with pytest.raises(ValueError, match="exactly one"):
        parse_circuit_input(circuit_qasm="OPENQASM 2.0;", circuit_preset="bell", num_qubits=None, qaoa_layers=None)


def test_parse_circuit_input_rejects_neither():
    from chatqec_mcp_tools.tools.ibm_hardware._circuits import parse_circuit_input

    with pytest.raises(ValueError, match="exactly one"):
        parse_circuit_input(circuit_qasm=None, circuit_preset=None, num_qubits=None, qaoa_layers=None)


def test_bell_preset_is_two_qubits():
    from chatqec_mcp_tools.tools.ibm_hardware._circuits import parse_circuit_input

    spec = parse_circuit_input(circuit_qasm=None, circuit_preset="bell", num_qubits=None, qaoa_layers=None)

    assert spec.num_logical_qubits == 2
    assert spec.circuit.count_ops().get("cx", 0) == 1


def test_ghz_preset_respects_num_qubits():
    from chatqec_mcp_tools.tools.ibm_hardware._circuits import parse_circuit_input

    spec = parse_circuit_input(circuit_qasm=None, circuit_preset="ghz", num_qubits=5, qaoa_layers=None)

    assert spec.num_logical_qubits == 5
    assert spec.circuit.count_ops().get("cx", 0) == 4


def test_qaoa_preset_layers_multiply_two_qubit_gate_count():
    from chatqec_mcp_tools.tools.ibm_hardware._circuits import parse_circuit_input

    one_layer = parse_circuit_input(circuit_qasm=None, circuit_preset="qaoa", num_qubits=4, qaoa_layers=1)
    two_layers = parse_circuit_input(circuit_qasm=None, circuit_preset="qaoa", num_qubits=4, qaoa_layers=2)

    assert two_layers.circuit.count_ops().get("cx", 0) == 2 * one_layer.circuit.count_ops().get("cx", 0)


def test_build_interaction_graph_weights_repeated_pairs():
    from chatqec_mcp_tools.tools.ibm_hardware._circuits import (
        build_interaction_graph,
        parse_circuit_input,
    )

    spec = parse_circuit_input(circuit_qasm=None, circuit_preset="ghz", num_qubits=3, qaoa_layers=None)
    interaction = build_interaction_graph(spec.circuit)

    assert set(interaction.graph.nodes) == {0, 1, 2}
    assert interaction.graph.has_edge(0, 1)
    assert interaction.graph.has_edge(0, 2)


def test_qaoa_layers_zero_produces_empty_circuit():
    from chatqec_mcp_tools.tools.ibm_hardware._circuits import parse_circuit_input

    spec = parse_circuit_input(circuit_qasm=None, circuit_preset="qaoa", num_qubits=4, qaoa_layers=0)

    # 0 layers should produce a circuit with no QAOA construction gates (h, cx, rz, rx)
    ops = spec.circuit.count_ops()
    assert ops.get("h", 0) == 0
    assert ops.get("cx", 0) == 0
    assert ops.get("rz", 0) == 0
    assert ops.get("rx", 0) == 0
