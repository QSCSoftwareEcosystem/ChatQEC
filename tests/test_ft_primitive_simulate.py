"""ft_primitive_simulate: FTPrimitiveBench primitives + noise families, decoded with PyMatching."""

import builtins
import hashlib

import pytest

from chatqec_mcp_tools.tools.ft_primitive_simulate import (
    FtPrimitiveInput,
    ft_primitive_simulate,
)

fpb = pytest.importorskip("ft_primitive_bench")


def _sha(circuit) -> str:
    return hashlib.sha256(str(circuit).encode()).hexdigest()


def test_memory_circuit_is_exactly_the_benchmark_circuit():
    """Parity: the tool must not rebuild the circuit; it must be FTPrimitiveBench's own."""
    from ft_primitive_bench.noise_models import uniform_depolarizing
    from ft_primitive_bench.surface_code.circuits import memory

    expected = uniform_depolarizing(p=1e-3).noisy_circuit(memory(3, 3, 3, meas_basis="Z"))
    out = ft_primitive_simulate(
        FtPrimitiveInput(primitive="memory", distance=3, rounds=3, p=1e-3, shots=200, seed=1)
    )
    assert out.ok, out.error_message
    assert out.circuit_sha256 == _sha(expected)
    assert out.num_qubits == expected.num_qubits
    assert out.num_detectors == expected.num_detectors


def test_same_seed_gives_the_same_result():
    inp = FtPrimitiveInput(primitive="memory", distance=3, rounds=3, p=5e-3, shots=2000, seed=7)
    a, b = ft_primitive_simulate(inp), ft_primitive_simulate(inp)
    assert a.ok and a.num_failures == b.num_failures


@pytest.mark.parametrize("primitive", ["memory", "transversal_h", "lattice_surgery", "s_gate"])
def test_every_primitive_builds_and_decodes(primitive):
    out = ft_primitive_simulate(
        FtPrimitiveInput(primitive=primitive, distance=3, p=1e-3, shots=200, seed=3)
    )
    assert out.ok, out.error_message
    assert 0.0 <= out.logical_error_rate <= 1.0
    assert out.ci95_low <= out.logical_error_rate <= out.ci95_high
    assert out.num_detectors > 0 and out.num_observables >= 1


@pytest.mark.parametrize(
    "noise",
    [
        {"noise": "pauli_biased", "bias_axis": "Z", "bias_factor": 10.0},
        {"noise": "measurement_biased", "bias_factor": 5.0},
        {"noise": "nonuniform", "sigma": 0.3},
    ],
)
def test_noise_families_change_the_circuit(noise):
    base = ft_primitive_simulate(
        FtPrimitiveInput(primitive="memory", distance=3, p=1e-3, shots=100, seed=2)
    )
    other = ft_primitive_simulate(
        FtPrimitiveInput(primitive="memory", distance=3, p=1e-3, shots=100, seed=2, **noise)
    )
    assert other.ok, other.error_message
    assert other.circuit_sha256 != base.circuit_sha256
    assert noise["noise"] in other.noise_model


def test_rectangular_patch_uses_both_distances():
    out = ft_primitive_simulate(
        FtPrimitiveInput(primitive="memory", x_distance=3, z_distance=5, p=1e-3, shots=100)
    )
    assert out.ok and out.x_distance == 3 and out.z_distance == 5


def test_s_gate_needs_a_square_patch():
    out = ft_primitive_simulate(
        FtPrimitiveInput(primitive="s_gate", x_distance=3, z_distance=5, p=1e-3, shots=10)
    )
    assert not out.ok and "square" in out.error_message


def test_missing_extra_gives_install_hint(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("ft_primitive_bench"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    out = ft_primitive_simulate(FtPrimitiveInput(primitive="memory", distance=3, shots=10))
    assert not out.ok and "ftprimitive" in out.error_message


def test_wilson_interval_is_exact_at_the_edges():
    from chatqec_mcp_tools.tools.ft_primitive_simulate import _wilson

    assert _wilson(0, 200)[0] == 0.0  # float error gave ~1.7e-18 > the observed rate 0
    assert _wilson(200, 200)[1] == 1.0
    lo, hi = _wilson(10, 1000)
    assert lo < 0.01 < hi
