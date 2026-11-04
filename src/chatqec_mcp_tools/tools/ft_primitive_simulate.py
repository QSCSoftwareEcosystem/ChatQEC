"""ft_primitive_simulate tool -- surface-code FT primitives under structured noise (FTPrimitiveBench).

Builds a logical primitive with FTPrimitiveBench's own circuit builders (memory, transversal
Hadamard, lattice surgery, logical S gate), applies one of its noise families (uniform,
Pauli-biased, measurement-biased, non-uniform), samples it with Stim and decodes with PyMatching,
the decoder the FTPrimitiveBench paper (arXiv:2605.04049) uses. The circuit is the benchmark's,
not a re-implementation: ``circuit_sha256`` lets a caller check that against the library.

The logical error rate counts shots in which ANY logical observable was predicted wrongly.
Needs the ``[ftprimitive]`` extra (FTPrimitiveBench pinned to commit b75fbd1).
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from chatqec_mcp_tools.tools._isolated_decode import _stdout_to_stderr

_MAX_SHOTS = 50_000
_INSTALL_HINT = "uv tool install 'chatqec-mcp-tools[ftprimitive]'"
_Z95 = 1.959963984540054


class FtPrimitiveInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    primitive: Literal["memory", "transversal_h", "lattice_surgery", "s_gate"] = Field(
        "memory",
        description=(
            "memory: repeated stabilizer rounds on one patch; transversal_h: rounds, one "
            "transversal Hadamard layer, rounds; lattice_surgery: two patches merged and split "
            "(MZZ for meas_basis Z, MXX for X); s_gate: logical S via ZZ surgery with a "
            "teleported Y-magic-state measurement"
        ),
    )
    distance: int = Field(3, ge=2, le=11, description="Code distance for a square patch")
    x_distance: int | None = Field(None, ge=2, le=11, description="X distance (default: distance)")
    z_distance: int | None = Field(None, ge=2, le=11, description="Z distance (default: distance)")
    rounds: int | None = Field(None, ge=1, le=33, description="memory: rounds (default: distance)")
    pre_rounds: int | None = Field(None, ge=1, le=33, description="Rounds before the operation")
    post_rounds: int | None = Field(None, ge=1, le=33, description="Rounds after the operation")
    merge_rounds: int | None = Field(None, ge=1, le=33, description="Surgery merge rounds")
    bridge_length: int = Field(1, ge=1, le=5, description="Surgery bridge between the patches")
    boundary_rounds: int = Field(1, ge=1, le=33, description="s_gate: boundary rounds")
    meas_basis: Literal["Z", "X"] = Field("Z", description="Final logical measurement basis")
    noise: Literal["uniform", "pauli_biased", "measurement_biased", "nonuniform"] = Field(
        "uniform", description="FTPrimitiveBench noise family"
    )
    p: float = Field(0.001, gt=0, le=0.05, description="Baseline physical error rate")
    bias_axis: Literal["Z", "X"] = Field("Z", description="pauli_biased: favoured Pauli axis")
    bias_factor: float = Field(
        1.0, ge=1.0, le=1000.0, description="pauli/measurement_biased: bias (1 = unbiased)"
    )
    sigma: float = Field(0.0, ge=0.0, le=1.0, description="nonuniform: relative scatter")
    shots: int = Field(2_000, ge=1, le=_MAX_SHOTS, description="Shots to sample")
    seed: int | None = Field(None, description="Seed for sampling and non-uniform noise")


class FtPrimitiveOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    primitive: str = ""
    x_distance: int = 0
    z_distance: int = 0
    rounds: dict[str, int] = Field(default_factory=dict)
    meas_basis: str = ""
    noise_model: str = ""
    decoder: str = "pymatching"
    logical_error_rate: float = 0.0
    std_error: float = 0.0
    ci95_low: float = 0.0
    ci95_high: float = 0.0
    shots: int = 0
    num_failures: int = 0
    num_qubits: int = 0
    num_detectors: int = 0
    num_observables: int = 0
    circuit_sha256: str = ""
    source: str = "FTPrimitiveBench (arXiv:2605.04049), commit b75fbd1"
    notes: str = ""
    error_message: str = ""


def _fail(message: str) -> FtPrimitiveOutput:
    return FtPrimitiveOutput(ok=False, error_message=message)


def _wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval for k failures in n shots."""
    p = k / n
    den = 1 + _Z95**2 / n
    centre = (p + _Z95**2 / (2 * n)) / den
    half = _Z95 * math.sqrt(p * (1 - p) / n + _Z95**2 / (4 * n * n)) / den
    low = 0.0 if k == 0 else max(0.0, centre - half)  # exact at the edges, no float residue
    high = 1.0 if k == n else min(1.0, centre + half)
    return low, high


def _build(inp: FtPrimitiveInput, xd: int, zd: int) -> tuple[Any, dict[str, int]]:
    from ft_primitive_bench.surface_code.circuits import (  # noqa: PLC0415
        lattice_surgery,
        memory,
        s_gate,
        transversal_h,
    )

    d = max(xd, zd)
    pre, post = inp.pre_rounds or d, inp.post_rounds or d
    if inp.primitive == "memory":
        r = {"rounds": inp.rounds or d}
        return memory(xd, zd, r["rounds"], meas_basis=inp.meas_basis), r
    if inp.primitive == "transversal_h":
        r = {"pre_rounds": pre, "post_rounds": post}
        return transversal_h(xd, zd, pre, post, meas_basis=inp.meas_basis), r
    merge = inp.merge_rounds or d
    if inp.primitive == "lattice_surgery":
        r = {"pre_rounds": pre, "merge_rounds": merge, "post_rounds": post}
        circuit = lattice_surgery(
            xd, zd, inp.bridge_length, pre, merge, post, meas_basis=inp.meas_basis
        )
        return circuit, r
    if xd != zd:
        raise ValueError("s_gate needs a square patch (x_distance == z_distance)")
    r = {
        "pre_rounds": pre,
        "merge_rounds": merge,
        "boundary_rounds": inp.boundary_rounds,
        "post_rounds": post,
    }
    circuit = s_gate(
        distance=d,
        bridge_length=inp.bridge_length,
        pre_rounds=pre,
        merge_rounds=merge,
        boundary_rounds=inp.boundary_rounds,
        post_rounds=post,
    )
    return circuit, r


def _noise(inp: FtPrimitiveInput) -> tuple[Any, str]:
    from ft_primitive_bench.noise_models import (  # noqa: PLC0415
        measurement_biased,
        nonuniform,
        pauli_biased,
        uniform_depolarizing,
    )

    if inp.noise == "pauli_biased":
        desc = f"pauli_biased(p={inp.p}, axis={inp.bias_axis}, bias_factor={inp.bias_factor})"
        return pauli_biased(inp.p, axis=inp.bias_axis, bias_factor=inp.bias_factor), desc
    if inp.noise == "measurement_biased":
        desc = f"measurement_biased(p={inp.p}, bias_factor={inp.bias_factor})"
        return measurement_biased(inp.p, bias_factor=inp.bias_factor), desc
    if inp.noise == "nonuniform":
        desc = f"nonuniform(p={inp.p}, sigma={inp.sigma}, seed={inp.seed})"
        return nonuniform(inp.p, sigma=inp.sigma, seed=inp.seed), desc
    return uniform_depolarizing(p=inp.p), f"uniform(p={inp.p})"


def _run(inp: FtPrimitiveInput) -> FtPrimitiveOutput:
    import numpy as np  # noqa: PLC0415
    import pymatching  # noqa: PLC0415

    xd, zd = inp.x_distance or inp.distance, inp.z_distance or inp.distance
    clean, rounds = _build(inp, xd, zd)
    model, desc = _noise(inp)
    noisy = model.noisy_circuit(clean)
    # Biased families emit PAULI_CHANNEL_2 with unequal components, which Stim only folds
    # into a detector error model as approximately disjoint errors.
    dem = noisy.detector_error_model(
        decompose_errors=True,
        ignore_decomposition_failures=True,
        approximate_disjoint_errors=True,
    )
    matcher = pymatching.Matching.from_detector_error_model(dem)
    det, obs = noisy.compile_detector_sampler(seed=inp.seed).sample(
        inp.shots, separate_observables=True
    )
    predicted = np.asarray(matcher.decode_batch(det))
    failures = int((predicted != obs).any(axis=1).sum())
    rate = failures / inp.shots
    low, high = _wilson(failures, inp.shots)
    return FtPrimitiveOutput(
        ok=True,
        primitive=inp.primitive,
        x_distance=xd,
        z_distance=zd,
        rounds=rounds,
        meas_basis=inp.meas_basis,
        noise_model=desc,
        logical_error_rate=rate,
        std_error=math.sqrt(rate * (1 - rate) / inp.shots),
        ci95_low=low,
        ci95_high=high,
        shots=inp.shots,
        num_failures=failures,
        num_qubits=noisy.num_qubits,
        num_detectors=noisy.num_detectors,
        num_observables=noisy.num_observables,
        circuit_sha256=hashlib.sha256(str(noisy).encode()).hexdigest(),
    )


def ft_primitive_simulate(inp: FtPrimitiveInput) -> FtPrimitiveOutput:
    try:
        with _stdout_to_stderr():
            return _run(inp)
    except ImportError:
        return _fail(f"FTPrimitiveBench is not installed: {_INSTALL_HINT}")
    except ValueError as exc:
        return _fail(str(exc))
    except Exception as exc:  # an FTPrimitiveBench/Stim construction or decoding failure
        return _fail(f"ft_primitive_simulate error: {type(exc).__name__}: {exc}")
