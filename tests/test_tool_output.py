"""Streamlit tool-output SVG sizing — the height helper is a pure function."""

from chatqec_app.components.tool_output import _svg_render_height


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
