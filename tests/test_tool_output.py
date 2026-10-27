"""Streamlit tool-output SVG sizing — the height helper is a pure function."""

from chatqec_app.components.tool_output import (
    _code_params_rows,
    _decoder_rows,
    _deltakit_rows,
    _ft_compile_rows,
    _guppy_rows,
    _pauli_overhead_rows,
    _svg_render_height,
)


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


def test_pauli_overhead_rows_pair_before_and_after_and_add_discard_rate():
    data = {
        "ok": True,
        "overhead": {
            "qubits_before": 3,
            "qubits_after": 4,
            "gates_before": 3,
            "gates_after": 12,
            "depth_before": 4,
            "depth_after": 11,
        },
        "cost_gamma": 1.07,
        "postselection_discard_rate": 0.0668,
    }
    rows = _pauli_overhead_rows(data)
    assert rows[0] == {"metric": "qubits", "before": 3, "after": 4}
    assert {"metric": "depth", "before": 4, "after": 11} in rows
    assert rows[-1] == {"metric": "postselection discard rate", "before": None, "after": 0.0668}


def test_pauli_overhead_rows_omit_discard_rate_when_not_estimated():
    data = {
        "ok": True,
        "overhead": {
            "qubits_before": 3,
            "qubits_after": 4,
            "gates_before": 3,
            "gates_after": 12,
            "depth_before": 4,
            "depth_after": 11,
        },
        "postselection_discard_rate": None,
    }
    assert all(r["metric"] != "postselection discard rate" for r in _pauli_overhead_rows(data))


def test_ft_compile_rows_for_clifford_t_report_t_count_first():
    data = {
        "ok": True,
        "mode": "clifford_t",
        "t_count": 65,
        "clifford_count": 84,
        "depth": 120,
        "num_qubits": 1,
    }
    assert _ft_compile_rows(data) == [
        {"metric": "T count", "value": 65},
        {"metric": "Clifford gates", "value": 84},
        {"metric": "depth", "value": 120},
        {"metric": "qubits", "value": 1},
    ]


def test_ft_compile_rows_for_pbc_report_rotations_and_no_t_count():
    data = {
        "ok": True,
        "mode": "pbc",
        "t_count": None,
        "clifford_count": None,
        "num_rotations": 6,
        "depth": 6,
        "num_qubits": 2,
    }
    rows = _ft_compile_rows(data)
    assert rows[0] == {"metric": "Pauli product rotations", "value": 6}
    assert all(r["metric"] not in ("T count", "Clifford gates") for r in rows)


def test_deltakit_rows_lead_with_the_error_rate_and_name_the_decoder_and_noise():
    data = {
        "ok": True,
        "code_family": "rotated_planar",
        "distance": 3,
        "rounds": 3,
        "decoder": "pymatching",
        "noise_model": "ToyNoise(p=0.002, p_measurement_flip=0.002)",
        "logical_error_rate": 0.0016,
        "std_error": 0.0003,
        "shots": 20000,
        "num_failures": 32,
        "num_qubits": 17,
        "num_detectors": 24,
        "num_observables": 1,
    }
    rows = _deltakit_rows(data)
    assert rows[0] == {"metric": "logical error rate", "value": 0.0016}
    assert {"metric": "decoder", "value": "pymatching"} in rows
    assert {"metric": "noise", "value": "ToyNoise(p=0.002, p_measurement_flip=0.002)"} in rows
    assert {"metric": "qubits", "value": 17} in rows


def test_guppy_rows_list_outcomes_by_frequency_with_fractions():
    data = {"ok": True, "registers": ["c0", "c1"], "shots": 200, "counts": {"1,1": 93, "0,0": 107}}
    rows = _guppy_rows(data)
    assert [r["outcome"] for r in rows] == ["0,0", "1,1"]
    assert rows[0] == {"outcome": "0,0", "registers": "c0,c1", "shots": 107, "fraction": 0.535}


def test_guppy_rows_are_empty_when_there_are_no_counts():
    assert _guppy_rows({"ok": True, "registers": [], "shots": 0, "counts": {}}) == []


def test_code_params_rows_show_the_code_its_source_and_where_it_came_from():
    data = {
        "ok": True,
        "rows": [
            {"n": 30, "k": 4, "d": 5, "source": "arXiv:2503.22071", "locator": "Table 1, weight 5"},
            {"n": 48, "k": 4, "d": 7, "source": "arXiv:2503.22071", "locator": "Table 1, weight 5"},
        ],
    }
    assert _code_params_rows(data) == [
        {"code": "[[30, 4, 5]]", "source": "arXiv:2503.22071", "where": "Table 1, weight 5"},
        {"code": "[[48, 4, 7]]", "source": "arXiv:2503.22071", "where": "Table 1, weight 5"},
    ]


def test_code_params_rows_are_empty_for_a_stim_result():
    assert _code_params_rows({"ok": True, "n": 9, "k": 1, "stim_code_distance": 3}) == []
