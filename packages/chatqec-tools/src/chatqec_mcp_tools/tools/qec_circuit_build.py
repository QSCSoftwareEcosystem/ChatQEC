"""qec_circuit_build tool - build a stim.Circuit for a QEC code family that plain
Stim doesn't ship built-in (toric code, color code, BB codes, PQRM), by delegating
to LightStim (https://github.com/QuTone/LightStim), pinned via the optional
`lightstim` extra. Requires `pip install chatqec-mcp-tools[lightstim]` (or
`uv sync --extra lightstim`); returns `ok=False` with a clear error_message when
lightstim isn't installed, rather than raising.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_MIN_D, _MAX_D = 3, 15

# Matches stim_diagram.py's _MAX_CIRCUIT_BYTES input cap, so a build-then-diagram
# follow-up call never gets refused purely because qec_circuit_build's own output
# was too large to hand it. circuit_text size tracks qubit count (roughly ~150
# bytes/qubit for these families) and, past the first couple of rounds, is
# essentially flat in `rounds`: Stim represents repeated syndrome-extraction
# rounds as a single compact `REPEAT N { ... }` block, so growing N barely
# grows the source text.
_MAX_CIRCUIT_TEXT_BYTES = 64 * 1024


class QecCircuitBuildInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code_family: Literal["toric_code", "color_code_666", "bb_code", "pqrm"]
    distance: int = Field(
        ...,
        ge=_MIN_D,
        le=_MAX_D,
        description=(
            f"Code distance in [{_MIN_D}, {_MAX_D}]. Valid values depend on "
            "code_family: toric_code accepts any distance in that range; "
            "color_code_666 requires an odd distance; bb_code only supports "
            "distance in {6, 10, 12} (discrete presets, not a continuous range); "
            "pqrm only supports distance in {3, 7} (discrete presets)."
        ),
    )
    rounds: int = Field(..., ge=1, le=200)
    protocol: Literal["memory"] = "memory"
    p: float = Field(
        0.0,
        ge=0.0,
        le=0.1,
        description=(
            "Uniform circuit-level noise rate (Stim's generated-circuit model: "
            "depolarizing after gates, flips at measure/reset). 0 = noiseless; set "
            "p > 0 before decoding or estimating logical error rates."
        ),
    )


class QecCircuitBuildOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    circuit_text: str = ""
    num_qubits: int = 0
    num_detectors: int = 0
    num_observables: int = 0
    error_message: str = ""


def qec_circuit_build(inp: QecCircuitBuildInput) -> QecCircuitBuildOutput:
    from chatqec_mcp_tools.lightstim_adapter import (  # noqa: PLC0415
        LightStimUnavailable,
        build_circuit,
    )

    try:
        circuit = build_circuit(
            code_family=inp.code_family,
            distance=inp.distance,
            rounds=inp.rounds,
            protocol=inp.protocol,
        )
    except LightStimUnavailable as e:
        return QecCircuitBuildOutput(ok=False, error_message=str(e))
    except (ValueError, RuntimeError) as e:
        # Covers both genuine lightstim runtime errors and this adapter's own
        # domain/work-ceiling validation (unsupported distance/preset/protocol,
        # or estimated work over the ceiling) — the message from either source
        # is already self-describing, so it's passed through as-is rather than
        # re-labeled as a "lightstim error".
        return QecCircuitBuildOutput(ok=False, error_message=f"could not build circuit: {e}")
    except (ImportError, TypeError, AttributeError) as e:
        # The availability gate only imports the top-level `lightstim` package;
        # real API calls happen via later submodule imports inside the builder
        # functions (e.g. lightstim.qec_code.color_code, lightstim.ir.qec_system),
        # executed after that gate passes. A partial install, or upstream renaming
        # something (a real risk -- LightStim is a young, fast-moving project;
        # this whole adapter module exists to isolate that risk), surfaces here
        # as ImportError/AttributeError/TypeError rather than ValueError/RuntimeError.
        # Labeled distinctly so "lightstim's API changed under us" is
        # distinguishable from "your input was invalid".
        return QecCircuitBuildOutput(ok=False, error_message=f"lightstim API error: {e}")

    if inp.p > 0:
        from chatqec_mcp_tools.tools._noise import add_uniform_noise  # noqa: PLC0415

        circuit = add_uniform_noise(circuit, inp.p)
    circuit_text = str(circuit)
    circuit_text_bytes = len(circuit_text.encode("utf-8"))
    if circuit_text_bytes > _MAX_CIRCUIT_TEXT_BYTES:
        return QecCircuitBuildOutput(
            ok=False,
            error_message=(
                f"circuit_text too large ({circuit_text_bytes:,} bytes > "
                f"{_MAX_CIRCUIT_TEXT_BYTES:,} byte cap); reduce distance and/or rounds"
            ),
        )

    return QecCircuitBuildOutput(
        ok=True,
        circuit_text=circuit_text,
        num_qubits=circuit.num_qubits,
        num_detectors=circuit.num_detectors,
        num_observables=circuit.num_observables,
    )
