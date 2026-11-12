"""iqm_backend_calibration tool -- per-qubit and per-coupler quality metrics of the IQM
device, from the server's current quality-metric and calibration sets."""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, ConfigDict, Field

from chatqec_mcp_tools.tools.iqm_hardware._client import IQMCredentialsError
from chatqec_mcp_tools.tools.iqm_hardware._hardware_graph import get_snapshot


def _us(seconds: float | None) -> float | None:
    return None if seconds is None else float(seconds) * 1e6


def _ns(seconds: float | None) -> float | None:
    return None if seconds is None else float(seconds) * 1e9


class IqmQubitCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    t1_us: float | None
    t2_echo_us: float | None
    t2_star_us: float | None
    readout_fidelity: float | None
    readout_error_0_to_1: float | None
    readout_error_1_to_0: float | None
    prx_fidelity: float | None


class IqmCouplerCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    qubits: list[str]
    native_gate: str = "cz"
    cz_fidelity: float | None  # interleaved RB
    clifford_cz_fidelity: float | None  # two-qubit Clifford RB
    cz_duration_ns: float | None


class IqmBackendCalibrationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IqmBackendCalibrationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    backend: str = ""
    dut_label: str = ""
    calibration_set_id: str = ""
    fetched_at: str = ""
    age_seconds: int = 0
    stale: bool = False
    qubits: list[IqmQubitCalibration] = Field(default_factory=list)
    couplers: list[IqmCouplerCalibration] = Field(default_factory=list)
    error_message: str = ""


def iqm_backend_calibration(
    inp: IqmBackendCalibrationInput,
) -> IqmBackendCalibrationOutput:
    try:
        cache = get_snapshot()
        snap = cache.payload
        qubits = [
            IqmQubitCalibration(
                name=q["name"],
                t1_us=_us(q.get("t1_s")),
                t2_echo_us=_us(q.get("t2_echo_s")),
                t2_star_us=_us(q.get("t2_s")),
                readout_fidelity=q.get("readout_fidelity"),
                readout_error_0_to_1=q.get("readout_error_0_to_1"),
                readout_error_1_to_0=q.get("readout_error_1_to_0"),
                prx_fidelity=q.get("prx_fidelity"),
            )
            for q in snap["qubits"]
        ]
        couplers = [
            IqmCouplerCalibration(
                qubits=list(c["qubits"]),
                cz_fidelity=c.get("cz_fidelity"),
                clifford_cz_fidelity=c.get("clifford_cz_fidelity"),
                cz_duration_ns=_ns(c.get("cz_duration_s")),
            )
            for c in snap["couplers"]
        ]
    except IQMCredentialsError as exc:
        return IqmBackendCalibrationOutput(ok=False, error_message=str(exc))
    except Exception as exc:
        return IqmBackendCalibrationOutput(
            ok=False, error_message=f"Failed to fetch IQM calibration: {exc}"
        )

    return IqmBackendCalibrationOutput(
        ok=True,
        backend=snap["quantum_computer"],
        dut_label=snap["dut_label"],
        calibration_set_id=snap["calibration_set_id"],
        fetched_at=dt.datetime.fromtimestamp(cache.fetched_at, tz=dt.UTC).isoformat(),
        age_seconds=int(cache.age_seconds),
        stale=cache.stale,
        qubits=qubits,
        couplers=couplers,
    )


__all__ = [
    "IqmBackendCalibrationInput",
    "IqmBackendCalibrationOutput",
    "iqm_backend_calibration",
]
