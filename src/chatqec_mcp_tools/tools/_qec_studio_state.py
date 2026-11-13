"""QEC Studio session state: .deq gadget parsing and the URL-fragment codec.

A QEC Studio link carries the whole session in its fragment, the way the app writes it:
``#1`` + base64url(raw DEFLATE(JSON)) with the JSON ``{v:1, deq, graph, nodes, edges,
components, sweep, sliders, viewports, settings}`` (see qec_studio.py).
"""

from __future__ import annotations

import base64
import json
import os
import re
import zlib
from typing import Any

_BASE = "https://bloqade.quera.com/studio/qec/"
_BLOCK = re.compile(r"^\s*(GADGET|COMPOSE)\s+([A-Za-z_]\w*)\s*\{", re.MULTILINE)


def studio_base() -> str:
    return os.environ.get("CHATQEC_QEC_STUDIO_URL", _BASE)


# ---------------------------------------------------------------- program parsing


def parse_gadgets(source: str) -> dict[str, dict[str, Any]]:
    """GADGET/COMPOSE name -> {inputs, outputs, readout} from the program text."""
    out: dict[str, dict[str, Any]] = {}
    matches = list(_BLOCK.finditer(source))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(source)
        body = source[m.end() : end]
        lines = [ln.strip() for ln in body.splitlines()]
        out[m.group(2)] = {
            "kind": m.group(1).lower(),
            "inputs": sum(1 for ln in lines if ln.startswith("INPUT ")),
            "outputs": sum(1 for ln in lines if ln.startswith("OUTPUT ")),
            "readout": any(ln.startswith("READOUT") for ln in lines),
        }
    return out


# ---------------------------------------------------------------- link encoding


def encode_state(state: dict[str, Any]) -> str:
    raw = json.dumps(state, separators=(",", ":")).encode()
    co = zlib.compressobj(9, zlib.DEFLATED, -15)  # raw DEFLATE, as fflate.deflateSync
    packed = co.compress(raw) + co.flush()
    return studio_base() + "#1" + base64.urlsafe_b64encode(packed).decode().rstrip("=")


def decode_state(url: str) -> dict[str, Any]:
    frag = url.split("#", 1)[1]
    if not frag.startswith("1"):
        raise ValueError("not a QEC Studio state link")
    data = frag[1:] + "=" * (-len(frag[1:]) % 4)
    return dict(json.loads(zlib.decompress(base64.urlsafe_b64decode(data), -15)))


def state(
    source: str,
    rf_nodes: list[dict[str, Any]],
    rf_edges: list[dict[str, Any]],
    graph: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    nodes = [
        {
            **{k: v for k, v in n.items() if k != "position"},
            "x": round(n["position"]["x"]),
            "y": round(n["position"]["y"]),
        }
        for n in rf_nodes
    ]
    extra = extra or {}
    return {
        "v": 1,
        "deq": source,
        "graph": graph,
        "nodes": nodes,
        "edges": rf_edges,
        "components": extra.get("components", {}),
        "sweep": extra.get("sweepConfig", {}),
        "sliders": extra.get("sliderIndices", {}),
        "viewports": extra.get("viewports", {}),
        "settings": None,  # app defaults (decoder, simulator, shots)
    }


def chain_state(
    source: str, chain: list[str], gadgets: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Session for a straight chain of single-port gadgets, ending in a Mean of the
    last gadget's readout."""
    step, y = 130, 175
    rf_nodes: list[dict[str, Any]] = []
    rf_edges: list[dict[str, Any]] = []
    g_nodes: list[dict[str, Any]] = []
    g_wires: list[dict[str, Any]] = []
    for i, name in enumerate(chain):
        info = gadgets[name]
        nid = f"g{i}"
        rf_nodes.append(
            {
                "id": nid,
                "type": "gadget",
                "position": {"x": 60 + i * step, "y": y},
                "data": {
                    "label": name,
                    "nIn": info["inputs"],
                    "nOut": info["outputs"],
                    "hasReadout": info["readout"],
                    "state": "idle",
                    "detectors": 0,
                    "measurements": 0,
                },
            }
        )
        g_nodes.append({"id": nid, "kind": "gadget", "gadget": name, "x": 60 + i * step, "y": y})
        if i:
            wid = f"w{i}"
            rf_edges.append(
                {
                    "id": wid,
                    "source": f"g{i - 1}",
                    "sourceHandle": "0",
                    "target": nid,
                    "targetHandle": "0",
                    "type": "quantum",
                    "data": {},
                }
            )
            g_wires.append({"id": wid, "from": [f"g{i - 1}", 0], "to": [nid, 0], "kind": "quantum"})
    if gadgets[chain[-1]]["readout"]:
        rf_nodes.append(
            {
                "id": "mean",
                "type": "mean",
                "position": {"x": 60 + len(chain) * step, "y": y + 85},
                "data": {"value": None, "shots": 0},
            }
        )
        rf_edges.append(
            {
                "id": "readout",
                "source": f"g{len(chain) - 1}",
                "sourceHandle": "bit",
                "target": "mean",
                "targetHandle": "bit_in",
                "type": "analysis",
                "data": {},
            }
        )
    # fit the chain into the canvas (about 1150 x 900 px beside the editor at 1500 px)
    width = 60 + (len(chain) + 1) * step
    zoom = min(1.5, 1000 / width)
    viewport = {"x": (1150 - width * zoom) / 2, "y": 450 - (y + 40) * zoom, "zoom": zoom}
    return state(
        source,
        rf_nodes,
        rf_edges,
        {"nodes": g_nodes, "wires": g_wires},
        {"viewports": {"main": viewport}},
    )


__all__ = ["chain_state", "decode_state", "encode_state", "parse_gadgets", "state", "studio_base"]
