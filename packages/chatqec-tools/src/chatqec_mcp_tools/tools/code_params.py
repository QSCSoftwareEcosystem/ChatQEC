"""code_params tool - report a named QEC code's parameters.

Stim families (surface, repetition, color): [[n, k]] and the stabilizer-generator count from
each family's standard formula, plus the code distance computed by Stim
(shortest_graphlike_error). No qiskit-qec dependency. For color codes the graphlike distance is
a lower bound on the true distance (hyperedge errors are undercounted).

Literature families (bb5, tile, directional_tile, barbell): the [[n, k, d]] values reported in
the papers, each with its source and location (see _paper_codes.py). Nothing is computed.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from chatqec_mcp_tools.tools._paper_codes import FAMILIES, paper_codes

_FAMILY_MAP = {
    "surface_code_rotated": "surface_code:rotated_memory_z",
    "repetition_code": "repetition_code:memory",
    "color_code": "color_code:memory_xyz",
}
_ODD_ONLY = {"surface_code_rotated", "color_code"}
_MIN_D, _MAX_D = 3, 15
_MAX_PAPER_D = 64


def _n_physical(family: str, d: int) -> int:
    if family == "surface_code_rotated":
        return d * d
    if family == "repetition_code":
        return d
    return (3 * d * d + 1) // 4  # triangular 6.6.6 color code


class CodeParamsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code_family: Literal[
        "surface_code_rotated",
        "repetition_code",
        "color_code",
        "bb5",
        "tile",
        "directional_tile",
        "barbell",
    ] = Field(
        ...,
        description=(
            "surface_code_rotated, repetition_code, color_code: computed with Stim. "
            "bb5, tile, directional_tile, barbell: parameters reported in the literature, "
            "returned with their source; nothing is computed"
        ),
    )
    distance: int | None = Field(
        default=None,
        ge=1,
        le=_MAX_PAPER_D,
        description=(
            f"Code distance. Required for the Stim families ({_MIN_D}-{_MAX_D}); for the "
            "literature families it filters the listed codes to that distance"
        ),
    )

    @model_validator(mode="after")
    def _distance_rules(self) -> CodeParamsInput:
        if self.code_family not in FAMILIES:
            if self.distance is None:
                raise ValueError(f"distance is required for {self.code_family}")
            if not _MIN_D <= self.distance <= _MAX_D:
                raise ValueError(
                    f"distance must be between {_MIN_D} and {_MAX_D} for Stim families"
                )
        return self


class CodeParamsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    code_family: str = ""
    distance: int = 0
    n: int = 0
    k: int = 0
    num_stabilizers: int = 0
    stim_code_distance: int = 0
    num_detectors: int = 0
    num_observables: int = 0
    notes: str = ""
    source: str = ""
    rows: list[dict[str, int | str]] = Field(default_factory=list)
    error_message: str = ""


def _literature_params(inp: CodeParamsInput) -> CodeParamsOutput:
    codes = paper_codes(inp.code_family)
    if inp.distance is not None:
        found = [c for c in codes if c.d == inp.distance]
        if not found:
            reported = ", ".join(str(d) for d in sorted({c.d for c in codes}))
            return CodeParamsOutput(
                ok=False,
                error_message=(
                    f"the papers report no {inp.code_family} code with distance "
                    f"{inp.distance}; reported distances: {reported}"
                ),
            )
        codes = found
    one = codes[0] if len(codes) == 1 else None
    return CodeParamsOutput(
        ok=True,
        code_family=inp.code_family,
        distance=one.d if one else (inp.distance or 0),
        n=one.n if one else 0,
        k=one.k if one else 0,
        notes="values copied from the papers cited in each row, not computed here",
        source=", ".join(sorted({c.source for c in codes})),
        rows=[
            {"n": c.n, "k": c.k, "d": c.d, "source": c.source, "locator": c.locator} for c in codes
        ],
    )


def code_params(inp: CodeParamsInput) -> CodeParamsOutput:
    if inp.code_family in FAMILIES:
        return _literature_params(inp)
    assert inp.distance is not None  # enforced by CodeParamsInput for the Stim families
    if inp.code_family in _ODD_ONLY and inp.distance % 2 == 0:
        return CodeParamsOutput(
            ok=False,
            error_message=f"{inp.code_family} requires odd distance; got {inp.distance}",
        )

    try:
        import stim  # noqa: PLC0415

        circuit = stim.Circuit.generated(
            _FAMILY_MAP[inp.code_family],
            distance=inp.distance,
            rounds=inp.distance,
            after_clifford_depolarization=0.001,
            before_measure_flip_probability=0.001,
            after_reset_flip_probability=0.001,
        )
        stim_distance = len(circuit.shortest_graphlike_error())
    except (ValueError, RuntimeError) as e:
        return CodeParamsOutput(ok=False, error_message=f"stim error: {e}")

    n = _n_physical(inp.code_family, inp.distance)
    k = 1
    notes = ""
    if inp.code_family == "color_code":
        notes = (
            "stim_code_distance is the graphlike distance (a lower bound); "
            "color-code hyperedge errors are undercounted"
        )
    return CodeParamsOutput(
        ok=True,
        code_family=inp.code_family,
        distance=inp.distance,
        n=n,
        k=k,
        num_stabilizers=n - k,
        stim_code_distance=stim_distance,
        num_detectors=circuit.num_detectors,
        num_observables=circuit.num_observables,
        notes=notes,
    )
