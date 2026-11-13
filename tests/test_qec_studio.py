"""QEC Studio tools with the network mocked out."""

from __future__ import annotations

import base64
import json
import zlib
from unittest.mock import patch

import pytest

from chatqec_mcp_tools.tools import qec_studio as qs
from chatqec_mcp_tools.tools._qec_studio_state import decode_state, encode_state, parse_gadgets

PROGRAM = """
CODE Rep [[3,1,3]] {
    LOGICAL X0*X1*X2 Z0*Z1*Z2
    STABILIZER Z0*Z1 Z1*Z2
}
GADGET PrepareZ {
    R 0 1 2
    OUTPUT Rep 0 1 2
}
GADGET Idle {
    INPUT Rep 0 2 4
    R 1 3
    CX 0 1 2 3
    CX 2 1 4 3
    M 1 3
    OUTPUT Rep 0 2 4
}
GADGET Merge {
    INPUT Rep 0 1 2
    INPUT Rep 3 4 5
    OUTPUT Rep 0 1 2
}
GADGET MeasureZ {
    INPUT Rep 0 1 2
    M 0 1 2
    READOUT rec[-3] rec[-2] rec[-1]
}
"""


def _compiled(src: str = PROGRAM) -> dict:
    return {"results": [{"id": 0, "rendered": src, "jitBase64": "", "runtime": "", "jitText": ""}]}


def test_parse_gadgets_ports_and_readout():
    g = parse_gadgets(PROGRAM)
    assert g["PrepareZ"] == {"kind": "gadget", "inputs": 0, "outputs": 1, "readout": False}
    assert g["Merge"]["inputs"] == 2
    assert g["MeasureZ"]["readout"] is True
    assert "Rep" not in g  # CODE blocks are not gadgets


def test_state_link_is_raw_deflate_base64url_with_version_prefix():
    url = encode_state({"v": 1, "deq": "x", "graph": {"nodes": [], "wires": []}})
    assert url.startswith("https://bloqade.quera.com/studio/qec/#1")
    frag = url.split("#1", 1)[1]
    assert "+" not in frag and "/" not in frag and "=" not in frag
    raw = base64.urlsafe_b64decode(frag + "=" * (-len(frag) % 4))
    assert json.loads(zlib.decompress(raw, -15))["deq"] == "x"
    assert decode_state(url)["v"] == 1


def test_link_builds_a_chain_with_mean():
    with patch.object(qs, "_compile", return_value=_compiled()):
        out = qs.qec_studio_link(
            qs.QecStudioLinkInput(program=PROGRAM, chain=["PrepareZ", "Idle", "Idle", "MeasureZ"])
        )
    assert out.ok, out.error_message
    assert out.has_mean
    st = decode_state(out.url)
    assert st["deq"] == PROGRAM
    assert [n["data"]["label"] for n in st["nodes"] if n["type"] == "gadget"] == [
        "PrepareZ", "Idle", "Idle", "MeasureZ",
    ]  # fmt: skip
    kinds = sorted(e["type"] for e in st["edges"])
    assert kinds == ["analysis", "quantum", "quantum", "quantum"]
    assert len(st["graph"]["wires"]) == 3
    assert st["viewports"]["main"]["zoom"] > 0


@pytest.mark.parametrize(
    ("chain", "message"),
    [
        (["PrepareZ", "Nope"], "Unknown gadget"),
        (["PrepareZ", "Merge", "MeasureZ"], "several code ports"),
        (["Idle", "MeasureZ"], "must start with a preparation"),
        (["PrepareZ", "MeasureZ", "Idle"], "no OUTPUT"),
    ],
)
def test_link_rejects_bad_chains(chain, message):
    with patch.object(qs, "_compile", return_value=_compiled()):
        out = qs.qec_studio_link(qs.QecStudioLinkInput(program=PROGRAM, chain=chain))
    assert not out.ok
    assert message in out.error_message


def test_compile_error_is_reported_not_raised():
    err = ValueError("INPUT port in GADGET 'X' references undefined CODE 'Nope'.")
    with patch.object(qs, "_compile", side_effect=err):
        out = qs.qec_studio_compile(qs.QecStudioCompileInput(program="GADGET X {\n}\n"))
    assert not out.ok
    assert "undefined CODE" in out.error_message


def test_compile_lists_gadgets_without_frame_helpers():
    rendered = PROGRAM + "\nGADGET frame__Rep {\n    INPUT Rep 0 1 2\n    OUTPUT Rep 0 1 2\n}\n"
    with patch.object(qs, "_compile", return_value=_compiled(rendered)):
        out = qs.qec_studio_compile(qs.QecStudioCompileInput(program=PROGRAM))
    assert out.ok
    assert [g.name for g in out.gadgets] == ["PrepareZ", "Idle", "Merge", "MeasureZ"]


def test_network_failure_is_ok_false():
    with patch.object(qs, "_compile", side_effect=RuntimeError("Could not reach QEC Studio")):
        out = qs.qec_studio_link(qs.QecStudioLinkInput(program=PROGRAM, chain=["PrepareZ"]))
    assert not out.ok
    assert "Could not reach" in out.error_message


_MANIFEST = [
    {"id": "noisy-idle", "title": "Memory Experiment", "description": "repetition-code memory",
     "source": "a/program.deq", "diagram": "a/diagram.deqviz"},
    {"id": "msd", "title": "MSD Repeat Until Success", "description": "magic state distillation",
     "source": "b/program.deq", "diagram": "b/diagram.deqviz"},
]  # fmt: skip


def _fake_get(path: str) -> bytes:
    if path.endswith(".deqviz"):
        node = {"id": "p", "type": "gadget", "position": {"x": 1.4, "y": 2.6}, "data": {}}
        return json.dumps({"graph": {"rfNodes": [node], "rfEdges": []}}).encode()
    return b"GADGET P {\n}\n"


def test_examples_without_query_list_all_without_links():
    with patch.object(qs, "_manifest", return_value=_MANIFEST), patch.object(qs, "_get") as get:
        out = qs.qec_studio_examples(qs.QecStudioExamplesInput())
    assert out.ok
    assert [e.id for e in out.examples] == ["noisy-idle", "msd"]
    assert all(e.url == "" for e in out.examples)
    get.assert_not_called()


def test_examples_query_returns_matches_with_links():
    with (
        patch.object(qs, "_manifest", return_value=_MANIFEST),
        patch.object(qs, "_get", side_effect=_fake_get),
    ):
        out = qs.qec_studio_examples(
            qs.QecStudioExamplesInput(query="magic state", include_source=True)
        )
    assert [e.id for e in out.examples] == ["msd"]
    st = decode_state(out.examples[0].url)
    assert st["nodes"][0]["x"] == 1 and st["nodes"][0]["y"] == 3
    assert out.examples[0].source.startswith("GADGET P")
