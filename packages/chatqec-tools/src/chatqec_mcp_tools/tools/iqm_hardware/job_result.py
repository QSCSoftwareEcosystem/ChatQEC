"""iqm_job_result tool -- status of an IQM job and, once it is done, its counts. Free."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from chatqec_mcp_tools.tools.iqm_hardware._client import IQMCredentialsError, call_worker


class IqmJobResultInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[0-9a-fA-F-]+$")


class IqmJobResultOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    job_id: str = ""
    status: str = ""
    counts: dict[str, int] = Field(default_factory=dict)
    error_message: str = ""


def iqm_job_result(inp: IqmJobResultInput) -> IqmJobResultOutput:
    try:
        reply = call_worker("job", job_id=inp.job_id)
    except IQMCredentialsError as exc:
        return IqmJobResultOutput(ok=False, error_message=str(exc))
    except Exception as exc:
        return IqmJobResultOutput(ok=False, error_message=f"Failed to fetch IQM job: {exc}")
    return IqmJobResultOutput(
        ok=True,
        job_id=inp.job_id,
        status=reply.get("status", ""),
        counts={str(k): int(v) for k, v in reply.get("counts", {}).items()},
    )


__all__ = ["IqmJobResultInput", "IqmJobResultOutput", "iqm_job_result"]
