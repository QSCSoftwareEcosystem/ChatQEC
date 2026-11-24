"""The vendored decoder-bench generators must produce the original code's circuits exactly.

tests/data/decoder_bench_reference_circuits.json was produced by decoder-bench itself at commit
9733a55 (its pinned requirements: stim 1.15.0, plus python-igraph 0.11.9) for every
circuit-noise memory config in its shipped results. A hash mismatch means the vendored copy,
the igraph shim, or a Stim version difference changed the circuit.
"""

import hashlib
import json
from pathlib import Path

import pytest

from chatqec_mcp_tools.tools._vendor.decoder_bench import generators as g

REF = json.loads(
    (Path(__file__).parent / "data" / "decoder_bench_reference_circuits.json").read_text()
)


def _build(cfg):
    if cfg["code"] == "surface":
        return g.gen_surface_circuit((cfg["d"], cfg["p"], cfg["basis"]))
    if cfg["code"] == "color":
        return g.gen_color_circuit((cfg["d"], cfg["p"]))
    return g.gen_qldpc_circuit((cfg["d"], cfg["p"], cfg["basis"]))


@pytest.mark.parametrize("name", sorted(REF))
def test_vendored_circuit_matches_original(name):
    cfg = REF[name]
    circuit = _build(cfg)
    assert (circuit.num_qubits, circuit.num_detectors, circuit.num_observables) == (
        cfg["num_qubits"], cfg["num_detectors"], cfg["num_observables"]
    )  # fmt: skip
    assert hashlib.sha256(str(circuit).encode()).hexdigest() == cfg["sha256"]
