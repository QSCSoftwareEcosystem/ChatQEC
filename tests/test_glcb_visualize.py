def test_glcb_registered():
    from chatqec_mcp_tools.server import build_server
    assert "glcb_visualize_url" in {t.name for t in build_server().list_tools()}


def test_bell_pair_matches_known_url_exactly():
    from chatqec_mcp_tools.tools.glcb_visualize import (
        GlcbGate, GlcbVisualizeInput, glcb_visualize_url,
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
        "https://gemini-logical-circuit-builder.vercel.app/"
        "#circuit={version:1.1,qubits:2,gates:[{gate:h,column:0,targets:[0]},"
        "{gate:cx,column:1,targets:[0,1]}],sampler:clifft}"
    )


def test_prep_and_postselect_matches_known_url_exactly():
    from chatqec_mcp_tools.tools.glcb_visualize import (
        GlcbGate, GlcbPostselect, GlcbPrep, GlcbVisualizeInput, glcb_visualize_url,
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
        GlcbGate, GlcbVisualizeInput, glcb_visualize_url,
    )
    out = glcb_visualize_url(
        GlcbVisualizeInput(qubits=2, gates=[GlcbGate(gate="h", column=0, targets=[5])])
    )
    assert out.ok is False
