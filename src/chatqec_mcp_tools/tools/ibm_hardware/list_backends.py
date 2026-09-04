"""ibm_list_backends tool -- backends visible to the configured IBM Quantum account."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from chatqec_mcp_tools.tools.ibm_hardware._cache import get_or_fetch
from chatqec_mcp_tools.tools.ibm_hardware._client import IBMCredentialsError, get_ibm_service

_DEFAULT_TTL_SECONDS = 60.0


class BackendInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    num_qubits: int
    operational: bool
    pending_jobs: int
    basis_gates: list[str]


class IbmListBackendsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IbmListBackendsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    backends: list[BackendInfo] = Field(default_factory=list)
    error_message: str = ""


def ibm_list_backends(inp: IbmListBackendsInput) -> IbmListBackendsOutput:
    ttl = float(os.environ.get("CHATQEC_IBM_BACKEND_LIST_TTL_SECONDS", _DEFAULT_TTL_SECONDS))

    def fetch() -> list[dict[str, Any]]:
        service = get_ibm_service()
        result = []
        for backend in service.backends():
            status = backend.status()
            result.append(
                {
                    "name": backend.name,
                    "num_qubits": backend.num_qubits,
                    "operational": status.operational,
                    "pending_jobs": status.pending_jobs,
                    "basis_gates": list(backend.operation_names),
                }
            )
        return result

    try:
        cache_result = get_or_fetch("list_backends", ttl, fetch)
    except IBMCredentialsError as exc:
        return IbmListBackendsOutput(ok=False, error_message=str(exc))
    except Exception as exc:  # network/SDK errors, never raise out of the tool
        return IbmListBackendsOutput(ok=False, error_message=f"Failed to list backends: {exc}")

    # Deserialization is guarded too, not just the fetch: cache_result.payload
    # may have come off disk, written by an older/newer code path (this key is
    # shared with map_circuit.py's auto-select) or hand-edited into a shape that
    # is still valid JSON but no longer matches BackendInfo. Left unguarded that
    # raised ValidationError straight out of the tool, breaking the feature-wide
    # "every failure path returns ok=False, nothing raises" contract.
    try:
        backends = [BackendInfo.model_validate(b) for b in cache_result.payload]
    except (ValidationError, KeyError, TypeError) as exc:
        return IbmListBackendsOutput(
            ok=False,
            error_message=(
                "Cached backend data is in an unexpected format; please retry "
                f"(a retry refetches from IBM Quantum). Details: {exc}"
            ),
        )

    return IbmListBackendsOutput(ok=True, backends=backends)
