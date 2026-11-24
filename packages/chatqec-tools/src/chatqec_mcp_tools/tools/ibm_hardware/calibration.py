# src/chatqec_mcp_tools/tools/ibm_hardware/calibration.py
"""ibm_backend_calibration tool -- T1/T2/readout/gate-error for one backend,
TTL-cached, always reports its own data age."""

from __future__ import annotations

import datetime as dt
import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from chatqec_mcp_tools.tools.ibm_hardware._cache import get_or_fetch
from chatqec_mcp_tools.tools.ibm_hardware._client import IBMCredentialsError, get_ibm_service
from chatqec_mcp_tools.tools.ibm_hardware._hardware_graph import build_hardware_graph

_DEFAULT_TTL_SECONDS = 3600.0


class QubitCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int
    t1_us: float | None
    t2_us: float | None
    readout_error: float | None


class CouplerCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    qubits: list[int]
    native_gate: str | None
    two_qubit_error: float | None
    gate_duration_ns: float | None


class IbmBackendCalibrationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: str


class IbmBackendCalibrationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    backend: str = ""
    fetched_at: str = ""
    age_seconds: int = 0
    stale: bool = False
    qubits: list[QubitCalibration] = Field(default_factory=list)
    couplers: list[CouplerCalibration] = Field(default_factory=list)
    error_message: str = ""


def ibm_backend_calibration(inp: IbmBackendCalibrationInput) -> IbmBackendCalibrationOutput:
    ttl = float(os.environ.get("CHATQEC_IBM_CALIBRATION_TTL_SECONDS", _DEFAULT_TTL_SECONDS))

    def fetch() -> dict[str, list[dict[str, Any]]]:
        service = get_ibm_service()
        backend = service.backend(inp.backend)
        hw_graph = build_hardware_graph(backend)
        qubits = [
            {
                "index": q,
                "t1_us": (d["T1"] * 1e6) if d.get("T1") is not None else None,
                "t2_us": (d["T2"] * 1e6) if d.get("T2") is not None else None,
                "readout_error": d.get("readout_error"),
            }
            for q, d in hw_graph.graph.nodes(data=True)
        ]
        couplers = [
            {
                "qubits": [u, v],
                "native_gate": d.get("native_gate"),
                "two_qubit_error": d.get("gate_error"),
                "gate_duration_ns": (d["gate_time"] * 1e9)
                if d.get("gate_time") is not None
                else None,
            }
            for u, v, d in hw_graph.graph.edges(data=True)
        ]
        return {"qubits": qubits, "couplers": couplers}

    try:
        cache_result = get_or_fetch(f"calibration/{inp.backend}", ttl, fetch)
    except IBMCredentialsError as exc:
        return IbmBackendCalibrationOutput(ok=False, error_message=str(exc))
    except Exception as exc:
        return IbmBackendCalibrationOutput(
            ok=False, error_message=f"Failed to fetch calibration for {inp.backend}: {exc}"
        )

    # Guarded for the same reason as list_backends.py: cache_result.payload may
    # have come off disk in a shape that is valid JSON but no longer matches
    # what this code reads (schema drift, a hand-edited file). Both the raw
    # ["qubits"]/["couplers"] subscripts and the model_validate calls sit inside
    # the guard, so a mismatched cache entry yields ok=False instead of a
    # KeyError/ValidationError escaping the tool.
    try:
        qubits = [QubitCalibration.model_validate(q) for q in cache_result.payload["qubits"]]
        couplers = [CouplerCalibration.model_validate(c) for c in cache_result.payload["couplers"]]
    except (ValidationError, KeyError, TypeError) as exc:
        return IbmBackendCalibrationOutput(
            ok=False,
            error_message=(
                f"Cached calibration data for {inp.backend} is in an unexpected format; "
                f"please retry (a retry refetches from IBM Quantum). Details: {exc}"
            ),
        )

    fetched_at_iso = dt.datetime.fromtimestamp(cache_result.fetched_at, tz=dt.UTC).isoformat()
    return IbmBackendCalibrationOutput(
        ok=True,
        backend=inp.backend,
        fetched_at=fetched_at_iso,
        age_seconds=int(cache_result.age_seconds),
        stale=cache_result.stale,
        qubits=qubits,
        couplers=couplers,
    )
