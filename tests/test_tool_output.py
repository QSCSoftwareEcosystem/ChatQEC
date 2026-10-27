"""Streamlit tool-output SVG sizing — the height helper is a pure function."""

from chatqec_app.components.tool_output import _decoder_rows, _svg_render_height


def test_height_from_viewbox_matches_aspect_ratio():
    # Stim timeline SVG: viewBox 1120x672 (aspect 0.6). At width 700 → ~440px.
    svg = '<svg viewBox="0 0 1120 672" version="1.1"></svg>'
    h = _svg_render_height(svg, width=700)
    assert 400 <= h <= 480  # not the old clipped 300


def test_height_caps_a_very_tall_svg():
    svg = '<svg viewBox="0 0 100 5000"></svg>'
    assert _svg_render_height(svg, width=700, cap=1200) == 1200


def test_height_falls_back_to_height_attr_without_viewbox():
    svg = '<svg width="400" height="360"></svg>'
    assert _svg_render_height(svg) == 380  # 360 + 20 padding


def test_height_default_when_no_dimensions():
    assert _svg_render_height("<svg></svg>", default=420) == 420


def test_decoder_rows_rank_successes_by_error_rate_and_put_failures_last():
    data = {
        "ok": True,
        "results": [
            {"decoder": "bposd", "ok": False, "note": "not installed: [decoders]"},
            {
                "decoder": "pymatching",
                "ok": True,
                "logical_error_rate": 0.01,
                "std_error": 0.002,
                "shots": 2000,
                "decode_seconds": 0.2,
                "note": "",
            },
            {
                "decoder": "tesseract",
                "ok": True,
                "logical_error_rate": 0.005,
                "std_error": 0.0016,
                "shots": 2000,
                "decode_seconds": 3.1,
                "note": "capped at 5,000 shots",
            },
        ],
    }
    rows = _decoder_rows(data)
    assert [r["decoder"] for r in rows] == ["tesseract", "pymatching", "bposd"]
    assert rows[0]["logical error rate"] == 0.005 and rows[0]["note"] == "capped at 5,000 shots"
    assert rows[2]["logical error rate"] is None and "not installed" in rows[2]["note"]
