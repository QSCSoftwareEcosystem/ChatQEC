"""stim_diagram tool - render a Stim circuit as SVG."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

_MAX_CIRCUIT_BYTES = 64 * 1024  # smaller cap - diagrams render slower


class StimDiagramInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    circuit: str = Field(..., description="Stim circuit source code")


class StimDiagramOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    svg: str = ""
    error_message: str = ""


def stim_diagram(inp: StimDiagramInput) -> StimDiagramOutput:
    if len(inp.circuit.encode("utf-8")) > _MAX_CIRCUIT_BYTES:
        return StimDiagramOutput(ok=False, error_message="circuit too large for diagram")

    try:
        import stim  # noqa: PLC0415

        circuit = stim.Circuit(inp.circuit)
        svg = str(circuit.diagram("timeline-svg"))
    except (ValueError, RuntimeError) as e:
        return StimDiagramOutput(ok=False, error_message=f"stim diagram error: {e}")

    return StimDiagramOutput(ok=True, svg=svg)
