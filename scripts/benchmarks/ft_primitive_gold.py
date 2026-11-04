"""Generate FTPrimitiveBench gold for the chatQEC evaluation.

Run in the tools repo's environment, which has FTPrimitiveBench pinned at commit b75fbd1:

    cd ~/chatqec-mcp-tools && uv run --extra ftprimitive python \\
        ~/chatQEC/scripts/benchmarks/ft_primitive_gold.py ~/chatQEC/eval/benchmarks/ft_primitive_bench/gold.jsonl

FTPrimitiveBench ships no results, so gold is computed here with the library itself (its
circuit builders and noise factories), sampled with Stim and decoded with PyMatching, the
decoder its paper uses. It deliberately does not import chatQEC's ft_primitive_simulate tool,
so the tool is checked against the library rather than against itself. Each item records the
noisy circuit's sha256: the tool must build the identical circuit.

Gold shots (200,000) are 10x an item's shots, so gold sampling error is small next to the
item's 95% interval. Non-uniform noise is realised with seed 42, the seed every item asks for.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pymatching
from ft_primitive_bench.noise_models import (
    measurement_biased,
    nonuniform,
    pauli_biased,
    uniform_depolarizing,
)
from ft_primitive_bench.surface_code.circuits import lattice_surgery, memory, s_gate, transversal_h

GOLD_SHOTS = 200_000
GOLD_SAMPLER_SEED = 20261104  # differs from the items' seed 42 on purpose
ITEM_SEED = 42

# (id, primitive, x_d, z_d, noise, params)  -- rounds default to max(x_d, z_d), bridge 1
GRID = [
    ("mem-d3-uniform-p3e-3", "memory", 3, 3, "uniform", {"p": 3e-3}),
    ("mem-d5-uniform-p3e-3", "memory", 5, 5, "uniform", {"p": 3e-3}),
    ("mem-d5-uniform-p5e-3", "memory", 5, 5, "uniform", {"p": 5e-3}),
    ("mem-d5-zbias10-p3e-3", "memory", 5, 5, "pauli_biased", {"p": 3e-3, "bias_axis": "Z", "bias_factor": 10.0}),
    ("mem-x3z5-zbias10-p3e-3", "memory", 3, 5, "pauli_biased", {"p": 3e-3, "bias_axis": "Z", "bias_factor": 10.0}),
    ("mem-x5z3-zbias10-p3e-3", "memory", 5, 3, "pauli_biased", {"p": 3e-3, "bias_axis": "Z", "bias_factor": 10.0}),
    ("mem-d5-measbias5-p3e-3", "memory", 5, 5, "measurement_biased", {"p": 3e-3, "bias_factor": 5.0}),
    ("mem-d5-nonuniform0.3-p3e-3", "memory", 5, 5, "nonuniform", {"p": 3e-3, "sigma": 0.3}),
    ("th-d3-uniform-p3e-3", "transversal_h", 3, 3, "uniform", {"p": 3e-3}),
    ("th-d5-uniform-p3e-3", "transversal_h", 5, 5, "uniform", {"p": 3e-3}),
    ("th-d5-zbias10-p3e-3", "transversal_h", 5, 5, "pauli_biased", {"p": 3e-3, "bias_axis": "Z", "bias_factor": 10.0}),
    ("th-x3z5-zbias10-p3e-3", "transversal_h", 3, 5, "pauli_biased", {"p": 3e-3, "bias_axis": "Z", "bias_factor": 10.0}),
    ("ls-d3-uniform-p3e-3", "lattice_surgery", 3, 3, "uniform", {"p": 3e-3}),
    ("ls-d5-uniform-p3e-3", "lattice_surgery", 5, 5, "uniform", {"p": 3e-3}),
    ("ls-d5-zbias10-p3e-3", "lattice_surgery", 5, 5, "pauli_biased", {"p": 3e-3, "bias_axis": "Z", "bias_factor": 10.0}),
    ("sg-d3-uniform-p3e-3", "s_gate", 3, 3, "uniform", {"p": 3e-3}),
    ("sg-d5-uniform-p3e-3", "s_gate", 5, 5, "uniform", {"p": 3e-3}),
]  # fmt: skip


def build(primitive: str, xd: int, zd: int):
    d = max(xd, zd)
    if primitive == "memory":
        return memory(xd, zd, d, meas_basis="Z")
    if primitive == "transversal_h":
        return transversal_h(xd, zd, d, d, meas_basis="Z")
    if primitive == "lattice_surgery":
        return lattice_surgery(xd, zd, 1, d, d, d, meas_basis="Z")
    return s_gate(
        distance=d, bridge_length=1, pre_rounds=d, merge_rounds=d, boundary_rounds=1, post_rounds=d
    )


def noise(kind: str, params: dict):
    p = params["p"]
    if kind == "pauli_biased":
        return pauli_biased(p, axis=params["bias_axis"], bias_factor=params["bias_factor"])
    if kind == "measurement_biased":
        return measurement_biased(p, bias_factor=params["bias_factor"])
    if kind == "nonuniform":
        return nonuniform(p, sigma=params["sigma"], seed=ITEM_SEED)
    return uniform_depolarizing(p=p)


def main(out: Path) -> None:
    rows = []
    for item_id, primitive, xd, zd, kind, params in GRID:
        circuit = noise(kind, params).noisy_circuit(build(primitive, xd, zd))
        dem = circuit.detector_error_model(
            decompose_errors=True,
            ignore_decomposition_failures=True,
            approximate_disjoint_errors=True,
        )
        matcher = pymatching.Matching.from_detector_error_model(dem)
        det, obs = circuit.compile_detector_sampler(seed=GOLD_SAMPLER_SEED).sample(
            GOLD_SHOTS, separate_observables=True
        )
        failures = int((np.asarray(matcher.decode_batch(det)) != obs).any(axis=1).sum())
        rate = failures / GOLD_SHOTS
        rows.append(
            {
                "id": item_id, "primitive": primitive, "x_distance": xd, "z_distance": zd,
                "noise": kind, **params, "gold_ler": rate,
                "gold_std_error": math.sqrt(rate * (1 - rate) / GOLD_SHOTS),
                "gold_shots": GOLD_SHOTS, "gold_sampler_seed": GOLD_SAMPLER_SEED,
                "circuit_sha256": hashlib.sha256(str(circuit).encode()).hexdigest(),
                "num_qubits": circuit.num_qubits, "num_detectors": circuit.num_detectors,
            }
        )  # fmt: skip
        print(item_id, f"LER={rate:.3e}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
