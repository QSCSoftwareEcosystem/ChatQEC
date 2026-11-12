"""iqm_list_backends tool -- the IQM quantum computer(s) behind the configured server."""

from __future__ import annotations

import datetime as dt
import os

from pydantic import BaseModel, ConfigDict, Field

from chatqec_mcp_tools.tools.iqm_hardware._client import IQMCredentialsError
from chatqec_mcp_tools.tools.iqm_hardware._hardware_graph import get_snapshot


class IqmBackendInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    dut_label: str
    num_qubits: int
    num_couplers: int
    healthy: bool | None
    native_gates: list[str]
    calibration_set_id: str


class IqmListBackendsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IqmListBackendsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    server_url: str = ""
    fetched_at: str = ""
    age_seconds: int = 0
    stale: bool = False
    backends: list[IqmBackendInfo] = Field(default_factory=list)
    note: str = ""
    error_message: str = ""


def iqm_list_backends(inp: IqmListBackendsInput) -> IqmListBackendsOutput:
    try:
        cache = get_snapshot()
        snap = cache.payload
        info = IqmBackendInfo(
            name=snap["quantum_computer"],
            dut_label=snap["dut_label"],
            num_qubits=len(snap["qubits"]),
            num_couplers=len(snap["couplers"]),
            healthy=snap.get("healthy"),
            native_gates=snap["native_gates"],
            calibration_set_id=snap["calibration_set_id"],
        )
    except IQMCredentialsError as exc:
        return IqmListBackendsOutput(ok=False, error_message=str(exc))
    except Exception as exc:
        return IqmListBackendsOutput(ok=False, error_message=f"Failed to query IQM: {exc}")

    return IqmListBackendsOutput(
        ok=True,
        server_url=os.environ.get("IQM_SERVER_URL", ""),
        fetched_at=dt.datetime.fromtimestamp(cache.fetched_at, tz=dt.UTC).isoformat(),
        age_seconds=int(cache.age_seconds),
        stale=cache.stale,
        backends=[info],
        note="The IQM server does not report queue depth; healthy=true means it accepts jobs.",
    )


__all__ = ["IqmListBackendsInput", "IqmListBackendsOutput", "iqm_list_backends"]
