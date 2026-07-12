"""threshold_sweep tool - sweep physical error rate p x code distance; MWPM-decode.

Generates memory-experiment circuits with Stim, decodes each with PyMatching, and
returns a table of logical error rates plus a log-log threshold plot (SVG).
"""

from __future__ import annotations

import io
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from chatqec_mcp_tools.tools._decoding import decode_circuit

_FAMILY_MAP = {
    "surface_code_rotated": "surface_code:rotated_memory_z",
    "repetition_code": "repetition_code:memory",
    "color_code": "color_code:memory_xyz",
}
_ODD_ONLY = {"surface_code_rotated", "color_code"}
_MIN_D, _MAX_D = 3, 15
_MAX_P = 0.5
_WORK_CEILING = 20_000_000
_MAX_SVG_BYTES = 100 * 1024


class SweepRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    distance: int
    p: float
    shots: int
    logical_error_rate: float


class ThresholdSweepInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code_family: Literal["surface_code_rotated", "repetition_code", "color_code"]
    distances: list[int] = Field(..., min_length=1, max_length=5)
    p_values: list[float] = Field(..., min_length=1, max_length=12)
    rounds: int | None = Field(None, description="Rounds per shot; defaults to distance")
    shots: int = Field(200, ge=1, le=1_000_000)


class ThresholdSweepOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    rows: list[SweepRow] = Field(default_factory=list)
    plot_svg: str = ""
    error_message: str = ""


def _validate(inp: ThresholdSweepInput) -> str | None:
    for d in inp.distances:
        if not (_MIN_D <= d <= _MAX_D):
            return f"distance {d} out of range [{_MIN_D}, {_MAX_D}]"
        if inp.code_family in _ODD_ONLY and d % 2 == 0:
            return f"{inp.code_family} requires odd distance; got {d}"
    for p in inp.p_values:
        if not (0.0 < p <= _MAX_P):
            return f"p={p} out of range (0, {_MAX_P}]"
    if inp.shots * len(inp.distances) * len(inp.p_values) > _WORK_CEILING:
        return f"work ceiling exceeded (> {_WORK_CEILING:,} decode-shots)"
    return None


def _render_plot(rows: list[SweepRow], family: str) -> str:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(5, 4))
    distances = sorted({r.distance for r in rows})
    for d in distances:
        pts = sorted((r.p, r.logical_error_rate) for r in rows if r.distance == d)
        ax.plot([p for p, _ in pts], [ler for _, ler in pts], marker="o", label=f"d={d}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("physical error rate p")
    ax.set_ylabel("logical error rate")
    ax.set_title(f"{family} threshold sweep")
    ax.legend()
    buf = io.StringIO()
    try:
        fig.savefig(buf, format="svg", bbox_inches="tight")
    finally:
        plt.close(fig)
    return buf.getvalue()


def threshold_sweep(inp: ThresholdSweepInput) -> ThresholdSweepOutput:
    err = _validate(inp)
    if err is not None:
        return ThresholdSweepOutput(ok=False, error_message=err)

    try:
        import stim  # noqa: PLC0415

        rows: list[SweepRow] = []
        for d in inp.distances:
            rounds = inp.rounds if inp.rounds is not None else d
            for p in inp.p_values:
                circuit = stim.Circuit.generated(
                    _FAMILY_MAP[inp.code_family],
                    distance=d,
                    rounds=rounds,
                    after_clifford_depolarization=p,
                    before_measure_flip_probability=p,
                    after_reset_flip_probability=p,
                )
                stats = decode_circuit(circuit, inp.shots)
                rows.append(SweepRow(
                    distance=d, p=p, shots=inp.shots,
                    logical_error_rate=stats.logical_error_rate,
                ))
    except (ValueError, RuntimeError) as e:
        return ThresholdSweepOutput(ok=False, error_message=f"sweep error: {e}")

    try:
        svg = _render_plot(rows, inp.code_family)
        if len(svg.encode("utf-8")) > _MAX_SVG_BYTES:
            svg = ""  # keep the table; drop an oversized plot rather than fail
    except Exception:  # plot is best-effort; a render failure must not fail the sweep
        svg = ""
    return ThresholdSweepOutput(ok=True, rows=rows, plot_svg=svg)
