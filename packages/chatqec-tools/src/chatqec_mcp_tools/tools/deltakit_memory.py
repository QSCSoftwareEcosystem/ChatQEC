"""deltakit_memory tool -- run a quantum-memory experiment with Riverlane's Deltakit.

Deltakit builds the code and its memory circuit, adds a noise model, and here decodes with
its LOCAL PyMatching decoder. Probed with deltakit 0.9.0 (the ``[deltakit]`` extra):

  - Local: code generation (rotated/unrotated planar, toric, repetition), the ToyNoise model,
    and ``PyMatchingDecoder``. The whole pipeline runs with no network access and no account.
  - Cloud only (through ``deltakit.Client``): the MWPM, BP+OSD, BeliefMatching, Local
    Clustering (the leakage-aware LCD), Ambiguity Clustering and Collision Clustering decoders,
    and leakage simulation. They are accepted as names and refused with "not configured", since
    this tool must never use a paid service.

We sample the detector error model ourselves, so ``seed`` makes a run reproducible. The logical
error rate counts shots in which ANY logical observable was predicted wrongly.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from chatqec_mcp_tools.tools._isolated_decode import _stdout_to_stderr

_MAX_SHOTS = 50_000
_CLOUD_DECODERS = ("mwpm", "belief_matching", "bposd", "lcd", "ac", "cc")
_INSTALL_HINT = "uv tool install 'chatqec-mcp-tools[deltakit]'"


class DeltakitMemoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code_family: Literal["rotated_planar", "unrotated_planar", "unrotated_toric", "repetition"] = (
        Field("rotated_planar", description="Deltakit code to build a memory experiment for")
    )
    distance: int = Field(3, ge=3, le=9, description="Code distance (width and height)")
    rounds: int | None = Field(None, ge=1, le=27, description="Syndrome rounds (default: distance)")
    logical_basis: Literal["Z", "X"] = Field("Z", description="Memory experiment basis")
    p: float = Field(0.001, gt=0, le=0.1, description="Two-qubit gate error rate (ToyNoise)")
    p_measurement_flip: float | None = Field(
        None, gt=0, le=0.1, description="Measurement flip probability (default: p)"
    )
    leakage_p: float = Field(
        0.0,
        ge=0,
        le=0.1,
        description="Leakage probability; needs the Deltakit cloud, not configured",
    )
    decoder: Literal["pymatching", "mwpm", "belief_matching", "bposd", "lcd", "ac", "cc"] = Field(
        "pymatching",
        description=(
            "pymatching runs locally. The others (mwpm, belief_matching, bposd, lcd = "
            "leakage-aware local clustering, ac, cc) are Deltakit cloud decoders and are not "
            "configured here"
        ),
    )
    shots: int = Field(2_000, ge=1, le=_MAX_SHOTS, description="Memory-experiment shots")
    seed: int | None = Field(default=None, description="Sampler seed for reproducibility")


class DeltakitMemoryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    code_family: str = ""
    distance: int = 0
    rounds: int = 0
    decoder: str = ""
    noise_model: str = ""
    logical_error_rate: float = 0.0
    std_error: float = 0.0
    shots: int = 0
    num_failures: int = 0
    num_qubits: int = 0
    num_detectors: int = 0
    num_observables: int = 0
    notes: str = ""
    error_message: str = ""


def _fail(message: str) -> DeltakitMemoryOutput:
    return DeltakitMemoryOutput(ok=False, error_message=message)


def _run(inp: DeltakitMemoryInput, rounds: int, flip: float) -> DeltakitMemoryOutput:
    import numpy as np  # noqa: PLC0415
    from deltakit.circuit.gates import PauliBasis  # noqa: PLC0415
    from deltakit.core.decoding_graphs import dem_to_decoding_graph_and_logicals  # noqa: PLC0415
    from deltakit.decode import PyMatchingDecoder  # noqa: PLC0415
    from deltakit.explorer import codes  # noqa: PLC0415
    from deltakit.explorer.qpu import QPU, ToyNoise  # noqa: PLC0415

    d = inp.distance
    code = {
        "rotated_planar": lambda: codes.RotatedPlanarCode(d, d),
        "unrotated_planar": lambda: codes.UnrotatedPlanarCode(d, d),
        "unrotated_toric": lambda: codes.UnrotatedToricCode(d, d),
        "repetition": lambda: codes.RepetitionCode(d),
    }[inp.code_family]()
    basis = PauliBasis.Z if inp.logical_basis == "Z" else PauliBasis.X
    circuit = codes.css_code_memory_circuit(code, num_rounds=rounds, logical_basis=basis)
    noise = ToyNoise(p=inp.p, p_measurement_flip=flip)
    noisy = QPU.from_circuit_compile_and_add_noise(circuit, noise_model=noise).as_stim_circuit()

    dem = noisy.detector_error_model(decompose_errors=True)
    graph, logicals = dem_to_decoding_graph_and_logicals(dem)
    decoder = PyMatchingDecoder(graph, logicals)
    det, obs = noisy.compile_detector_sampler(seed=inp.seed).sample(
        inp.shots, separate_observables=True
    )
    predicted = np.asarray(decoder.decode_batch_to_logical_flip(det))
    failures = int((predicted != obs).any(axis=1).sum())
    rate = failures / inp.shots
    return DeltakitMemoryOutput(
        ok=True,
        code_family=inp.code_family,
        distance=d,
        rounds=rounds,
        decoder="pymatching",
        noise_model=f"ToyNoise(p={inp.p}, p_measurement_flip={flip})",
        logical_error_rate=rate,
        std_error=math.sqrt(rate * (1 - rate) / inp.shots),
        shots=inp.shots,
        num_failures=failures,
        num_qubits=noisy.num_qubits,
        num_detectors=noisy.num_detectors,
        num_observables=noisy.num_observables,
    )


def deltakit_memory(inp: DeltakitMemoryInput) -> DeltakitMemoryOutput:
    if inp.decoder in _CLOUD_DECODERS:
        return _fail(
            f"the {inp.decoder} decoder runs in the Deltakit cloud, which is not configured "
            "here (it needs an account and can cost money); use decoder='pymatching'"
        )
    if inp.leakage_p > 0:
        return _fail(
            "leakage simulation and leakage-aware decoding need the Deltakit cloud, which is "
            "not configured here; run with leakage_p=0"
        )
    try:
        with _stdout_to_stderr():
            return _run(inp, inp.rounds or inp.distance, inp.p_measurement_flip or inp.p)
    except ImportError:
        return _fail(f"Deltakit is not installed: {_INSTALL_HINT}")
    except Exception as exc:  # a Deltakit/Stim construction or decoding failure
        return _fail(f"deltakit error: {type(exc).__name__}: {exc}")
