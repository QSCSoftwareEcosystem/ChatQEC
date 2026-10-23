"""decoder_compare tool - decode the SAME sampled syndromes of a Stim circuit with
several decoders and rank them by logical error rate.

Decoders:
  - ``pymatching``  MWPM baseline (base install).
  - ``tesseract``   Google quantumlib's search-based most-likely-error decoder;
                    strong on color codes and correlated/hyperedge noise.
  - ``bposd`` / ``bplsd`` / ``belief_find``  BP+OSD, BP+LSD, BP+union-find
                    from the ``ldpc`` package; the usual qLDPC decoders.
  - ``cudaq_relay_bp`` / ``cudaq_bposd``  NVIDIA CUDA-Q QEC GPU decoders
                    (RelayBP, BP+OSD); see _cudaq_decoders.py (``[cudaq]`` extra).

The CPU non-MWPM decoders come through QuEra's ``bloqade-decoders`` (``[decoders]``
extra) and run in a child process (_isolated_decode.py), because ldpc's BP+OSD
can segfault and would otherwise take the whole MCP server down.

Circuits: pass Stim text, or a ``code_family`` to build one here with uniform
circuit-level noise ``p``: Stim's built-ins, or LightStim's toric / 6.6.6 color /
BB (incl. the [[144,12,12]] gross code, bb_code d=12) / PQRM. Building here keeps
large circuits out of the LLM context (the noisy gross code is ~84 KB of text).

Measured on a d=5, 5-round, p=0.002 circuit-level-noise memory (2000 shots):
surface code  PM 0.0005 (0.01 s), Tesseract 0.0005 (3.1 s), BP+OSD-CS5 0.001 (1.1 s);
color code    PM 0.038, Tesseract 0.005 (7.9 s), BP+OSD-CS5 0.0075, BP+LSD 0.013,
              GPU RelayBP 0.005 (1.0 s).
Gross code (bb_code d=12, 3 rounds, p=0.005, 500 shots): GPU RelayBP 0.0 (5.3 s),
GPU BP+OSD 0.036 (1.0 s); the CPU ldpc decoders exceed the per-decoder timeout there.
Tesseract costs ~1.5-4 ms/shot at d=5, hence its separate shot cap below.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from chatqec_mcp_tools.tools._cudaq_decoders import CUDAQ_DECODERS, cudaq_predict
from chatqec_mcp_tools.tools._isolated_decode import OSD_CS_MAX_ERRORS, predict_isolated
from chatqec_mcp_tools.tools.threshold_sweep import _FAMILY_MAP, _ODD_ONLY

if TYPE_CHECKING:
    import numpy as np
    import stim
    from numpy.typing import NDArray

    BoolArray = NDArray[np.bool_]

DecoderName = Literal[
    "pymatching", "tesseract", "bposd", "bplsd", "belief_find", "cudaq_relay_bp", "cudaq_bposd"
]

_MAX_CIRCUIT_BYTES = 256 * 1024
_MAX_SHOTS = 50_000
_MAX_TESSERACT_SHOTS = 5_000  # ~3 ms/shot at d=5 keeps the call inside the 60 s MCP budget
_DECODER_TIMEOUT_S = 50.0  # per CPU decoder child process
_MIN_D, _MAX_D = 3, 15
_LIGHTSTIM_FAMILIES = ("toric_code", "color_code_666", "bb_code", "pqrm")
_MAX_P = 0.1


class DecoderCompareInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    circuit: str | None = Field(
        None, description="Stim circuit with DETECTORs and OBSERVABLE_INCLUDEs (or use code_family)"
    )
    code_family: (
        Literal[
            "surface_code_rotated",
            "repetition_code",
            "color_code",
            "toric_code",
            "color_code_666",
            "bb_code",
            "pqrm",
        ]
        | None
    ) = Field(
        None,
        description=(
            "Generate a noisy memory circuit instead of passing one. bb_code is "
            "bivariate bicycle (distance 6, 10 or 12; d=12 is IBM's [[144,12,12]] "
            "gross code); pqrm distance 3 or 7; others need an odd distance"
        ),
    )
    distance: int = Field(5, ge=_MIN_D, le=_MAX_D, description="Code distance for code_family")
    p: float = Field(0.001, gt=0, le=_MAX_P, description="Circuit-level noise rate for code_family")
    rounds: int | None = Field(
        None, ge=1, le=3 * _MAX_D, description="Syndrome rounds (default: distance)"
    )
    decoders: list[DecoderName] = Field(
        default=["pymatching", "tesseract", "bposd"],
        min_length=1,
        description=(
            "Decoders to compare on identical syndrome samples: pymatching (MWPM, CPU), "
            "tesseract (Google search decoder, CPU), bposd / bplsd / belief_find (ldpc, CPU), "
            "cudaq_relay_bp (NVIDIA CUDA-Q RelayBP, GPU), cudaq_bposd (NVIDIA CUDA-Q "
            "BP+OSD, GPU). Pick the cudaq_* names when the user asks for NVIDIA, CUDA-Q, "
            "or GPU decoding."
        ),
    )
    shots: int = Field(
        2_000,
        ge=1,
        le=_MAX_SHOTS,
        description=f"Shots (tesseract decodes at most {_MAX_TESSERACT_SHOTS:,})",
    )
    seed: int | None = Field(None, description="Sampler seed for reproducibility")

    @model_validator(mode="after")
    def _one_source(self) -> DecoderCompareInput:
        if (self.circuit is None) == (self.code_family is None):
            raise ValueError("pass exactly one of circuit or code_family")
        if self.code_family in (*_ODD_ONLY, "color_code_666") and self.distance % 2 == 0:
            raise ValueError(f"{self.code_family} needs an odd distance")
        return self


class DecoderResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decoder: str
    ok: bool
    logical_error_rate: float = 0.0
    std_error: float = 0.0
    num_errors: int = 0
    shots: int = 0
    decode_seconds: float = 0.0
    note: str = ""


class DecoderCompareOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    results: list[DecoderResult] = []
    best: str = ""
    num_detectors: int = 0
    num_observables: int = 0
    error_message: str = ""


def _pymatching_predict(circuit: stim.Circuit, det: BoolArray) -> tuple[BoolArray, str]:
    import pymatching  # noqa: PLC0415

    try:
        dem = circuit.detector_error_model(decompose_errors=True)
        note = ""
    except ValueError:
        # Hyperedges that Stim cannot split into graphlike pieces (e.g. color codes):
        # PyMatching then approximates them, which is the honest MWPM baseline.
        dem = circuit.detector_error_model(decompose_errors=False)
        note = "error model not graphlike; MWPM approximates hyperedges"
    return pymatching.Matching.from_detector_error_model(dem).decode_batch(det), note


def _run_one(
    name: str, circuit: stim.Circuit, dem: stim.DetectorErrorModel, det: BoolArray, obs: BoolArray
) -> DecoderResult:
    note = ""
    if name == "tesseract" and len(det) > _MAX_TESSERACT_SHOTS:
        det, obs = det[:_MAX_TESSERACT_SHOTS], obs[:_MAX_TESSERACT_SHOTS]
        note = f"capped at {_MAX_TESSERACT_SHOTS:,} shots"
    start = time.perf_counter()
    try:
        if name == "pymatching":
            pred, note = _pymatching_predict(circuit, det)
        elif name in CUDAQ_DECODERS:
            pred = cudaq_predict(name, dem, det)
        else:
            pred, child_seconds = predict_isolated(name, str(dem), det, _DECODER_TIMEOUT_S)
            start = time.perf_counter() - child_seconds  # report decode time, not fork time
            if name == "bposd" and dem.num_errors > OSD_CS_MAX_ERRORS:
                note = "large error model: OSD order 0 instead of OSD-CS 5"
    except ImportError:
        extra = "cudaq" if name in CUDAQ_DECODERS else "decoders"
        return DecoderResult(
            decoder=name,
            ok=False,
            note=f"not installed: uv tool install 'chatqec-mcp-tools[{extra}]'",
        )
    except (ValueError, RuntimeError) as e:
        return DecoderResult(decoder=name, ok=False, note=f"decode error: {e}")
    seconds = time.perf_counter() - start
    shots = len(det)
    errors = int((pred != obs).any(axis=1).sum())
    rate = errors / shots
    return DecoderResult(
        decoder=name,
        ok=True,
        logical_error_rate=rate,
        std_error=math.sqrt(rate * (1 - rate) / shots),
        num_errors=errors,
        shots=shots,
        decode_seconds=round(seconds, 4),
        note=note,
    )


def _build_circuit(inp: DecoderCompareInput) -> stim.Circuit:
    import stim  # noqa: PLC0415

    if inp.circuit is not None:
        return stim.Circuit(inp.circuit)
    if inp.code_family in _LIGHTSTIM_FAMILIES:
        from chatqec_mcp_tools.lightstim_adapter import build_circuit  # noqa: PLC0415
        from chatqec_mcp_tools.tools._noise import add_uniform_noise  # noqa: PLC0415

        noiseless = build_circuit(
            code_family=inp.code_family,
            distance=inp.distance,
            rounds=inp.rounds or inp.distance,
            protocol="memory",
        )
        return add_uniform_noise(noiseless, inp.p)
    return stim.Circuit.generated(
        _FAMILY_MAP[inp.code_family],  # type: ignore[index]
        distance=inp.distance,
        rounds=inp.rounds or inp.distance,
        after_clifford_depolarization=inp.p,
        before_measure_flip_probability=inp.p,
        after_reset_flip_probability=inp.p,
    )


def decoder_compare(inp: DecoderCompareInput) -> DecoderCompareOutput:
    if inp.circuit is not None and len(inp.circuit.encode("utf-8")) > _MAX_CIRCUIT_BYTES:
        return DecoderCompareOutput(
            ok=False, error_message=f"circuit too large (>{_MAX_CIRCUIT_BYTES // 1024} KB)"
        )
    try:
        circuit = _build_circuit(inp)
        dem = circuit.detector_error_model(decompose_errors=False)
    except (ValueError, RuntimeError) as e:
        return DecoderCompareOutput(ok=False, error_message=f"invalid circuit: {e}")
    except ImportError as e:  # LightStim family without the [lightstim] extra
        return DecoderCompareOutput(ok=False, error_message=f"missing extra: {e}")
    if dem.num_observables == 0:
        return DecoderCompareOutput(
            ok=False, error_message="circuit defines no logical observables"
        )
    if dem.num_errors == 0:
        return DecoderCompareOutput(
            ok=False,
            error_message=(
                "circuit has no noise, so there is nothing to decode; pass code_family "
                "with p, or use qec_circuit_build with p > 0"
            ),
        )

    sampler = circuit.compile_detector_sampler(seed=inp.seed)
    det, obs = sampler.sample(shots=inp.shots, separate_observables=True)
    results = [_run_one(name, circuit, dem, det, obs) for name in dict.fromkeys(inp.decoders)]
    ranked = sorted(
        (r for r in results if r.ok), key=lambda r: (r.logical_error_rate, r.decode_seconds)
    )
    return DecoderCompareOutput(
        ok=bool(ranked),
        results=results,
        best=ranked[0].decoder if ranked else "",
        num_detectors=dem.num_detectors,
        num_observables=dem.num_observables,
        error_message="" if ranked else "no decoder succeeded",
    )
