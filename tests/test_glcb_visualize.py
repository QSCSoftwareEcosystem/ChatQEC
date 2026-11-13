import pytest


def test_glcb_registered():
    from chatqec_mcp_tools.server import build_server

    assert "glcb_visualize_url" in {t.name for t in build_server().list_tools()}


def test_bell_pair_matches_known_url_exactly():
    from chatqec_mcp_tools.tools.glcb_visualize import (
        GlcbGate,
        GlcbVisualizeInput,
        glcb_visualize_url,
    )

    out = glcb_visualize_url(
        GlcbVisualizeInput(
            qubits=2,
            gates=[
                GlcbGate(gate="h", column=0, targets=[0]),
                GlcbGate(gate="cx", column=1, targets=[0, 1]),
            ],
        )
    )
    assert out.ok is True
    assert out.url == (
        "https://bloqade.quera.com/studio/gemini/"
        "#circuit={version:1.1,qubits:2,gates:[{gate:h,column:0,targets:[0]},"
        "{gate:cx,column:1,targets:[0,1]}],sampler:clifft}"
    )


def test_prep_and_postselect_matches_known_url_exactly():
    from chatqec_mcp_tools.tools.glcb_visualize import (
        GlcbGate,
        GlcbPostselect,
        GlcbPrep,
        GlcbVisualizeInput,
        glcb_visualize_url,
    )

    out = glcb_visualize_url(
        GlcbVisualizeInput(
            qubits=3,
            gates=[
                GlcbGate(gate="h", column=0, targets=[2]),
                GlcbGate(gate="cx", column=1, targets=[2, 1]),
                GlcbGate(gate="cx", column=2, targets=[0, 1]),
                GlcbGate(gate="h", column=3, targets=[0]),
                GlcbGate(gate="h", column=4, targets=[2]),
            ],
            prep=[GlcbPrep(qubit=0, theta=1.570796, phi=0, lam=0)],
            postselect=[GlcbPostselect(qubit=0, value=0), GlcbPostselect(qubit=1, value=0)],
        )
    )
    assert out.ok is True
    assert "prep:[{qubit:0,theta:1.570796,phi:0,lam:0}]" in out.url
    assert "postselect:[{qubit:0,value:0},{qubit:1,value:0}]" in out.url


def test_out_of_range_target_returns_ok_false():
    from chatqec_mcp_tools.tools.glcb_visualize import (
        GlcbGate,
        GlcbVisualizeInput,
        glcb_visualize_url,
    )

    out = glcb_visualize_url(
        GlcbVisualizeInput(qubits=2, gates=[GlcbGate(gate="h", column=0, targets=[5])])
    )
    assert out.ok is False


def test_malformed_gate_name_rejected_by_pydantic():
    """Regression test: a gate name containing fragment-grammar syntax
    (`}`, `,`, `:`) must fail pydantic validation instead of silently
    splicing extra keys (e.g. an overriding `qubits:999`) into the URL
    fragment with ok=True."""
    import pytest
    from pydantic import ValidationError

    from chatqec_mcp_tools.tools.glcb_visualize import GlcbGate

    with pytest.raises(ValidationError):
        GlcbGate(gate="h},qubits:999,x:{y", column=0, targets=[0])


def test_malformed_gadget_type_rejected_by_pydantic():
    import pytest
    from pydantic import ValidationError

    from chatqec_mcp_tools.tools.glcb_visualize import GlcbGadget

    with pytest.raises(ValidationError):
        GlcbGadget(type="T},x:{y", dataQubit=0, column=0)


def test_non_finite_prep_angle_rejected_by_pydantic():
    """Regression test: NaN/Infinity in theta/phi/lam previously escaped as
    an uncaught ValueError/OverflowError from `_num`'s `int(x)` call; must now
    fail cleanly at pydantic validation instead."""
    import math

    import pytest
    from pydantic import ValidationError

    from chatqec_mcp_tools.tools.glcb_visualize import GlcbPrep

    with pytest.raises(ValidationError):
        GlcbPrep(qubit=0, theta=math.inf)
    with pytest.raises(ValidationError):
        GlcbPrep(qubit=0, theta=math.nan)


def test_negative_target_rejected_by_pydantic():
    import pytest
    from pydantic import ValidationError

    from chatqec_mcp_tools.tools.glcb_visualize import GlcbGate

    with pytest.raises(ValidationError):
        GlcbGate(gate="h", column=0, targets=[-1])


def test_oversized_prep_list_rejected_by_pydantic():
    """Mirrors gates' existing _MAX_GATES cap: prep/postselect/gadgets/
    placement lists must also be bounded so a huge list can't produce an
    oversized URL."""
    import pytest
    from pydantic import ValidationError

    from chatqec_mcp_tools.tools.glcb_visualize import (
        _MAX_LIST_LEN,
        GlcbGate,
        GlcbPrep,
        GlcbVisualizeInput,
    )

    with pytest.raises(ValidationError):
        GlcbVisualizeInput(
            qubits=2,
            gates=[GlcbGate(gate="h", column=0, targets=[0])],
            prep=[GlcbPrep(qubit=0) for _ in range(_MAX_LIST_LEN + 1)],
        )


def test_t_is_rejected_as_a_gate_and_accepted_as_a_gadget():
    """Gemini Studio silently drops unknown gates, so `t` must fail validation."""
    from pydantic import ValidationError

    from chatqec_mcp_tools.tools.glcb_visualize import (
        GlcbGadget,
        GlcbGate,
        GlcbVisualizeInput,
        glcb_visualize_url,
    )

    with pytest.raises(ValidationError):
        GlcbGate(gate="t", column=0, targets=[0])
    out = glcb_visualize_url(
        GlcbVisualizeInput(
            qubits=1,
            gates=[GlcbGate(gate="h", column=0, targets=[0])],
            gadgets=[GlcbGadget(type="T", dataQubit=0, column=1)],
        )
    )
    assert out.ok
    assert "gadgets:[{type:T,dataQubit:0,column:1}]" in out.url
    assert out.url.startswith("https://bloqade.quera.com/studio/gemini/#circuit=")


def test_wrong_target_count_is_an_error_not_a_silently_dropped_gate():
    from chatqec_mcp_tools.tools.glcb_visualize import (
        GlcbGate,
        GlcbVisualizeInput,
        glcb_visualize_url,
    )

    out = glcb_visualize_url(
        GlcbVisualizeInput(qubits=2, gates=[GlcbGate(gate="cx", column=0, targets=[0])])
    )
    assert not out.ok
    assert "needs 2 target" in out.error_message


def test_star_gadget_angle_is_encoded():
    from chatqec_mcp_tools.tools.glcb_visualize import (
        GlcbGadget,
        GlcbVisualizeInput,
        glcb_visualize_url,
    )

    out = glcb_visualize_url(
        GlcbVisualizeInput(
            qubits=1, gates=[], gadgets=[GlcbGadget(type="Star", dataQubit=0, column=0, angle=0.25)]
        )
    )
    assert "{type:Star,dataQubit:0,column:0,angle:0.25}" in out.url
