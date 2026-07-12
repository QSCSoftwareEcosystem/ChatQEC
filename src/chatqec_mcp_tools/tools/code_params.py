"""code_params tool - report a named QEC code's parameters + a Stim-computed distance.

Reports [[n, k]] and the stabilizer-generator count from each family's standard
formula, plus the code distance computed by Stim (shortest_graphlike_error). No
qiskit-qec dependency. For color codes the graphlike distance is a lower bound on
the true distance (hyperedge errors are undercounted).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_FAMILY_MAP = {
    "surface_code_rotated": "surface_code:rotated_memory_z",
    "repetition_code": "repetition_code:memory",
    "color_code": "color_code:memory_xyz",
}
_ODD_ONLY = {"surface_code_rotated", "color_code"}
_MIN_D, _MAX_D = 3, 15


def _n_physical(family: str, d: int) -> int:
    if family == "surface_code_rotated":
        return d * d
    if family == "repetition_code":
        return d
    return (3 * d * d + 1) // 4  # triangular 6.6.6 color code


class CodeParamsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code_family: Literal["surface_code_rotated", "repetition_code", "color_code"]
    distance: int = Field(..., ge=_MIN_D, le=_MAX_D)


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
    error_message: str = ""


def code_params(inp: CodeParamsInput) -> CodeParamsOutput:
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
