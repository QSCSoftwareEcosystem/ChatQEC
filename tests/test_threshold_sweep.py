"""threshold_sweep tool tests."""

from chatqec_mcp_tools.tools.threshold_sweep import (
    ThresholdSweepInput,
    ThresholdSweepOutput,
    threshold_sweep,
)


def test_sweep_returns_rows_and_plot():
    out = threshold_sweep(ThresholdSweepInput(
        code_family="repetition_code",
        distances=[3, 5],
        p_values=[0.05, 0.1],
        shots=200,
    ))
    assert isinstance(out, ThresholdSweepOutput)
    assert out.ok is True
    assert len(out.rows) == 4  # |distances| * |p_values|
    for row in out.rows:
        assert 0.0 <= row.logical_error_rate <= 0.5
        assert row.shots == 200
    assert "<svg" in out.plot_svg


def test_sweep_rejects_work_ceiling():
    out = threshold_sweep(ThresholdSweepInput(
        code_family="repetition_code",
        distances=[3, 5, 7, 9, 11],
        p_values=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12],
        shots=1_000_000,
    ))
    assert out.ok is False
    assert "ceiling" in out.error_message.lower() or "too large" in out.error_message.lower()


def test_sweep_rejects_even_distance_for_surface_code():
    out = threshold_sweep(ThresholdSweepInput(
        code_family="surface_code_rotated",
        distances=[4],
        p_values=[0.01],
        shots=50,
    ))
    assert out.ok is False
    assert "odd" in out.error_message.lower()


def test_sweep_survives_plot_failure(monkeypatch):
    import chatqec_mcp_tools.tools.threshold_sweep as ts

    def _boom(*args, **kwargs):
        raise RuntimeError("render exploded")

    monkeypatch.setattr(ts, "_render_plot", _boom)
    out = threshold_sweep(ThresholdSweepInput(
        code_family="repetition_code", distances=[3], p_values=[0.1], shots=50,
    ))
    assert out.ok is True
    assert out.plot_svg == ""
    assert len(out.rows) == 1
