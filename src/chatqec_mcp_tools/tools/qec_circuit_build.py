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


class QecCircuitBuildInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code_family: Literal["toric_code", "color_code_666", "bb_code", "pqrm"]
    distance: int = Field(..., ge=_MIN_D, le=_MAX_D)
    rounds: int = Field(..., ge=1, le=200)
    protocol: str = "memory"


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
        return QecCircuitBuildOutput(ok=False, error_message=f"lightstim error: {e}")

    return QecCircuitBuildOutput(
        ok=True,
        circuit_text=str(circuit),
        num_qubits=circuit.num_qubits,
        num_detectors=circuit.num_detectors,
        num_observables=circuit.num_observables,
    )
