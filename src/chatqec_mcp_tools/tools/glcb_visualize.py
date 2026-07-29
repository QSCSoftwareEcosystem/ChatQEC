"""glcb_visualize_url tool - build a shareable QuEra GLCB
(gemini-logical-circuit-builder.vercel.app) visualization link for a small
circuit. Pure string encoding, no network call, no external dependency.

The app reads its state from the URL fragment in a relaxed, non-JSON,
JS-object-literal-like grammar: unquoted keys, unquoted bare-word string
values (gate names, sampler name, gadget type), numbers as-is. Verified by
round-tripping the app's own example URLs byte-for-byte before this tool
was written — do not "clean up" into real JSON, it will not parse.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

_BASE_URL = "https://gemini-logical-circuit-builder.vercel.app/"
_MAX_GATES = 200
_MAX_LIST_LEN = 200  # same cap as _MAX_GATES, applied to the other per-qubit
# lists (prep/postselect/gadgets/placement) so a huge list can't blow up the
# generated URL's size the same way an oversized `gates` list already can't.

# Gate/gadget-type names are encoded unquoted, bare-word, into the URL
# fragment (see module docstring) — anything containing `{`, `}`, `:`, `,`
# etc. would silently splice extra keys into the fragment instead of failing
# loudly. Restricting to identifier-like tokens makes that a pydantic
# validation error instead of a corrupted-but-`ok=True` URL.
_TOKEN_PATTERN = r"^[A-Za-z0-9_]+$"

_NonNegInt = Annotated[int, Field(ge=0)]


class GlcbGate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gate: str = Field(..., pattern=_TOKEN_PATTERN, description="Gate name, e.g. h, cx, s, t")
    column: int = Field(..., ge=0)
    targets: list[_NonNegInt] = Field(..., min_length=1)


class GlcbPrep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    qubit: int = Field(..., ge=0)
    theta: float = Field(0, allow_inf_nan=False)
    phi: float = Field(0, allow_inf_nan=False)
    lam: float = Field(0, allow_inf_nan=False)


class GlcbPostselect(BaseModel):
    model_config = ConfigDict(extra="forbid")
    qubit: int = Field(..., ge=0)
    value: int = Field(..., ge=0, le=1)


class GlcbGadget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str = Field(..., pattern=_TOKEN_PATTERN, description="e.g. T, Star")
    dataQubit: int = Field(..., ge=0)
    column: int


class GlcbPlacement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    qubit: int = Field(..., ge=0)
    slot: int = Field(..., ge=0)


class GlcbVisualizeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    qubits: int = Field(..., ge=1, le=64)
    gates: list[GlcbGate] = Field(..., max_length=_MAX_GATES)
    prep: list[GlcbPrep] = Field(default_factory=list, max_length=_MAX_LIST_LEN)
    postselect: list[GlcbPostselect] = Field(default_factory=list, max_length=_MAX_LIST_LEN)
    gadgets: list[GlcbGadget] = Field(default_factory=list, max_length=_MAX_LIST_LEN)
    placement: list[GlcbPlacement] = Field(default_factory=list, max_length=_MAX_LIST_LEN)
    sampler: Literal["clifft"] = "clifft"


class GlcbVisualizeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    url: str = ""
    error_message: str = ""


def _num(x: float) -> str:
    if isinstance(x, int) or (isinstance(x, float) and x == int(x)):
        return str(int(x))
    return repr(round(x, 6))


def _obj(d: dict[str, Any]) -> str:
    # Defensive: every caller today builds `d` via model_dump(exclude_none=True),
    # so no value is ever None in practice, but this guard is kept (with a
    # correct `is not None` check, not the `v not in (None,)` membership test
    # that a truthy/falsy-but-not-None value like `0` or `False` could also
    # match) in case a future caller passes an un-filtered dict.
    return "{" + ",".join(f"{k}:{_encode(v)}" for k, v in d.items() if v is not None) + "}"


def _encode(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return _num(v)
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return "[" + ",".join(_encode(x) for x in v) + "]"
    if isinstance(v, dict):
        return _obj(v)
    raise TypeError(type(v))


def glcb_visualize_url(inp: GlcbVisualizeInput) -> GlcbVisualizeOutput:
    for g in inp.gates:
        if any(t >= inp.qubits for t in g.targets):
            return GlcbVisualizeOutput(
                ok=False,
                error_message=f"gate {g.gate!r} targets qubit >= qubits={inp.qubits}",
            )

    spec: dict[str, Any] = {"version": 1.1, "qubits": inp.qubits}
    spec["gates"] = [g.model_dump(exclude_none=True) for g in inp.gates]
    if inp.prep:
        spec["prep"] = [p.model_dump(exclude_none=True) for p in inp.prep]
    if inp.postselect:
        spec["postselect"] = [p.model_dump(exclude_none=True) for p in inp.postselect]
    if inp.gadgets:
        spec["gadgets"] = [g.model_dump(exclude_none=True) for g in inp.gadgets]
    if inp.placement:
        spec["placement"] = [p.model_dump(exclude_none=True) for p in inp.placement]
    spec["sampler"] = inp.sampler

    fragment = _encode(spec)
    return GlcbVisualizeOutput(ok=True, url=f"{_BASE_URL}#circuit={fragment}")
