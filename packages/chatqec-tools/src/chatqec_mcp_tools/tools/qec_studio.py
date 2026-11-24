"""QuEra Bloqade QEC Studio tools (https://bloqade.quera.com/studio/qec/).

QEC Studio runs gadget-level QEC programs written in ``.deq``: Stim instructions plus
``CODE`` (a code's logicals and stabilizers), ``GADGET`` (a circuit block with INPUT /
OUTPUT code ports and an optional READOUT) and ``COMPOSE`` blocks, optionally templated
with Mako (``<% d = int(context.get('d', 3)) %>``). Three tools:

* ``qec_studio_examples``: the app's built-in examples (its ``examples/manifest.json``),
  each with a link that opens it.
* ``qec_studio_compile``: validate a program with the app's own compile service
  (``api/compile``, the endpoint the app calls); returns the gadgets or the compiler's
  error message.
* ``qec_studio_link``: compile a program, lay out a chain of its gadgets
  (prepare -> rounds -> measure -> Mean) and return a shareable link.

Links carry the whole session in the URL fragment, exactly as the app writes it itself:
``#1`` + base64url(raw DEFLATE(JSON {v:1, deq, graph, nodes, edges, ...})) (fflate's
deflateSync; checked 2026-11-13 by loading such a link headless and running Simulate).
The hosted app has no widget backend (``api/widgets/create`` returns 404), so the fragment
is the only way to share a session without QuEra's own deployment.
"""

from __future__ import annotations

import itertools
import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from chatqec_mcp_tools.tools._qec_studio_state import (
    chain_state,
    decode_state,
    encode_state,
    parse_gadgets,
    state,
    studio_base,
)

_UA = "chatqec-mcp-tools"
_TIMEOUT_S = 120.0
_MANIFEST_TTL_S = 24 * 3600.0
_MAX_SOURCE_CHARS = 100_000
_MAX_CHAIN = 60
_MAX_MATCHES = 3
_MAX_URL_CHARS = 6000


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word  # noqa: PLR2004


_manifest_cache: dict[str, Any] = {}


# ---------------------------------------------------------------- network helpers


def _get(path: str) -> bytes:
    req = urllib.request.Request(studio_base() + path, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
        return bytes(resp.read())


def _compile(source: str) -> dict[str, Any]:
    """POST the program to QEC Studio's compile service. Raises ValueError with the
    compiler's message on a 400 (a bad program), RuntimeError on other failures."""
    body = json.dumps({"source": source, "variants": [{"id": "", "makoDefs": {}}]}).encode()
    req = urllib.request.Request(
        studio_base() + "api/compile",
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": _UA},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            return dict(json.load(resp))
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", "replace")[:1500]
        if exc.code == 400:  # noqa: PLR2004
            raise ValueError(text) from exc
        raise RuntimeError(f"QEC Studio compile service returned {exc.code}: {text}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not reach QEC Studio: {exc}") from exc


def _manifest() -> list[dict[str, Any]]:
    now = time.time()
    if _manifest_cache and now - _manifest_cache["at"] < _MANIFEST_TTL_S:
        return list(_manifest_cache["items"])
    items = json.loads(_get("examples/manifest.json"))
    _manifest_cache.update(at=now, items=items)
    return list(items)


# ---------------------------------------------------------------- tools


class QecStudioExamplesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field("", max_length=200, description="Words to match in title/description")
    include_source: bool = Field(
        False, description="Also return each match's .deq program (needs a query)"
    )


class QecStudioExample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    title: str
    description: str
    url: str = ""
    source: str = ""
    note: str = ""


class QecStudioExamplesOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    examples: list[QecStudioExample] = Field(default_factory=list)
    studio_url: str = ""
    error_message: str = ""


def qec_studio_examples(inp: QecStudioExamplesInput) -> QecStudioExamplesOutput:
    try:
        items = _manifest()
    except Exception as exc:
        return QecStudioExamplesOutput(ok=False, error_message=f"Could not load examples: {exc}")
    words = [_stem(w) for w in re.findall(r"\w+", inp.query.lower()) if len(w) > 1]

    def score(e: dict[str, Any]) -> int:
        text = f"{e.get('id', '')} {e.get('title', '')} {e.get('description', '')}".lower()
        return sum(w in text for w in words)

    if words:  # only the best-scoring matches: links are 1-10 KB of base64 each
        best = max((score(e) for e in items), default=0)
        picked = [e for e in items if best and score(e) == best][:_MAX_MATCHES]
    else:
        picked = items
    out = []
    for e in picked:
        url, source = "", ""
        if not words:  # links and sources run 1-10 KB each: only for query matches
            out.append(
                QecStudioExample(
                    id=e.get("id", ""),
                    title=e.get("title", ""),
                    description=e.get("description", ""),
                )
            )
            continue
        try:  # a link needs the example's program and diagram
            source = _get(f"examples/{e['source']}").decode()
            diagram = json.loads(_get(f"examples/{e['diagram']}"))["graph"]
            url = encode_state(
                state(
                    source,
                    diagram["rfNodes"],
                    diagram["rfEdges"],
                    {"nodes": [], "wires": []},
                    diagram,
                )
            )  # the app rebuilds the engine graph from the diagram on load
        except Exception:  # the listing is still useful without a link
            url = ""
        note = ""
        if len(url) > _MAX_URL_CHARS:  # keeps the model's context small
            url, note = (
                "",
                (
                    f"Link too long to return: open QEC Studio ({studio_base()}) and pick "
                    f"'{e.get('title', '')}' in its Example gallery (shown on load)."
                ),
            )
        out.append(
            QecStudioExample(
                id=e.get("id", ""),
                title=e.get("title", ""),
                description=e.get("description", ""),
                url=url,
                source=source if inp.include_source else "",
                note=note,
            )
        )
    return QecStudioExamplesOutput(ok=True, examples=out, studio_url=studio_base())


class QecStudioCompileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    program: str = Field(..., min_length=1, max_length=_MAX_SOURCE_CHARS, description=".deq source")


class QecStudioGadget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    kind: str
    inputs: int
    outputs: int
    readout: bool


class QecStudioCompileOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    gadgets: list[QecStudioGadget] = Field(default_factory=list)
    rendered_program: str = ""  # after Mako templating
    error_message: str = ""


def qec_studio_compile(inp: QecStudioCompileInput) -> QecStudioCompileOutput:
    try:
        result = _compile(inp.program)["results"][0]
    except ValueError as exc:
        return QecStudioCompileOutput(ok=False, error_message=f"Compile error: {exc}")
    except Exception as exc:
        return QecStudioCompileOutput(ok=False, error_message=str(exc))
    rendered = str(result.get("rendered", ""))
    gadgets = [
        QecStudioGadget(name=k, **v)
        for k, v in parse_gadgets(rendered or inp.program).items()
        if not k.startswith("frame__")  # compiler-generated, hidden in the app too
    ]
    return QecStudioCompileOutput(ok=True, gadgets=gadgets, rendered_program=rendered[:20_000])


class QecStudioLinkInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    program: str = Field(..., min_length=1, max_length=_MAX_SOURCE_CHARS, description=".deq source")
    chain: list[str] = Field(
        ...,
        min_length=1,
        max_length=_MAX_CHAIN,
        description="Gadget names in order, e.g. ['PrepareZ', 'Idle', 'Idle', 'MeasureZ']. "
        "Each must have at most one INPUT and one OUTPUT; the first has no INPUT.",
    )


class QecStudioLinkOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    url: str = ""
    chain: list[str] = Field(default_factory=list)
    has_mean: bool = False
    note: str = ""
    error_message: str = ""


def _chain_error(chain: list[str], gadgets: dict[str, dict[str, Any]]) -> str:
    missing = [n for n in chain if n not in gadgets]
    if missing:
        return f"Unknown gadget(s) {missing}; the program defines {sorted(gadgets)}."
    multi = [n for n in chain if gadgets[n]["inputs"] > 1 or gadgets[n]["outputs"] > 1]
    if multi:
        return (
            f"Gadget(s) {sorted(set(multi))} have several code ports; this tool lays out a "
            "single-block chain only. Build multi-block layouts in the QEC Studio editor."
        )
    if gadgets[chain[0]]["inputs"]:
        return f"The chain must start with a preparation gadget (no INPUT); {chain[0]} has one."
    for a, b in itertools.pairwise(chain):
        if gadgets[a]["outputs"] != 1:
            return f"{a} has no OUTPUT, so nothing can follow it."
        if gadgets[b]["inputs"] != 1:
            return f"{b} has no INPUT, so it cannot follow {a}."
    return ""


def qec_studio_link(inp: QecStudioLinkInput) -> QecStudioLinkOutput:
    try:
        result = _compile(inp.program)["results"][0]
    except ValueError as exc:
        return QecStudioLinkOutput(ok=False, error_message=f"Compile error: {exc}")
    except Exception as exc:
        return QecStudioLinkOutput(ok=False, error_message=str(exc))
    gadgets = parse_gadgets(str(result.get("rendered", "")) or inp.program)
    err = _chain_error(inp.chain, gadgets)
    if err:
        return QecStudioLinkOutput(ok=False, error_message=err)
    session = chain_state(inp.program, inp.chain, gadgets)
    has_mean = bool(gadgets[inp.chain[-1]]["readout"])
    return QecStudioLinkOutput(
        ok=True,
        url=encode_state(session),
        chain=inp.chain,
        has_mean=has_mean,
        note="Open the link and press Simulate; the Mean block shows the logical readout."
        if has_mean
        else "The last gadget has no READOUT, so no Mean block was added.",
    )


__all__ = [
    "QecStudioCompileInput",
    "QecStudioCompileOutput",
    "QecStudioExamplesInput",
    "QecStudioExamplesOutput",
    "QecStudioLinkInput",
    "QecStudioLinkOutput",
    "chain_state",
    "decode_state",
    "encode_state",
    "parse_gadgets",
    "qec_studio_compile",
    "qec_studio_examples",
    "qec_studio_link",
]
