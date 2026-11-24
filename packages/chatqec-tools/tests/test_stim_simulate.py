"""stim_simulate tool tests."""

from chatqec_mcp_tools.tools.stim_simulate import (
    StimSimulateInput,
    StimSimulateOutput,
    stim_simulate,
)


def test_simulate_runs_simple_circuit():
    out = stim_simulate(
        StimSimulateInput(
            circuit="H 0\nCNOT 0 1\nM 0 1\n",
            shots=10,
        )
    )
    assert isinstance(out, StimSimulateOutput)
    assert out.ok is True
    assert out.shots == 10
    # M 0 and M 1 should each give a sample column
    assert "shape" in out.summary.lower() or "10" in out.summary


def test_simulate_returns_error_for_invalid_circuit():
    out = stim_simulate(
        StimSimulateInput(
            circuit="NOT_A_GATE 0\n",
            shots=4,
        )
    )
    assert out.ok is False
    assert "error" in (out.summary + out.error_message).lower()


def test_simulate_rejects_oversize_circuit():
    huge = "M 0\n" * 100_000  # >1 MB
    out = stim_simulate(
        StimSimulateInput(
            circuit=huge,
            shots=1,
        )
    )
    assert out.ok is False
    assert "too large" in out.error_message.lower()


def test_simulate_caps_shots():
    out = stim_simulate(
        StimSimulateInput(
            circuit="M 0\n",
            shots=10_000_000,
        )
    )
    # Either capped silently or rejected; behavior is deterministic
    assert out.ok is False or out.shots <= 1_000_000


def test_simulate_preview_shows_all_rows_when_shots_fit():
    out = stim_simulate(StimSimulateInput(circuit="H 0\nCNOT 0 1\nM 0 1\n", shots=10))
    assert len(out.samples_preview) == 10
    assert "ALL rows shown" in out.summary


def test_simulate_preview_flags_truncation_for_large_shots():
    out = stim_simulate(StimSimulateInput(circuit="M 0\n", shots=1000))
    assert len(out.samples_preview) == 20
    assert "TRUNCATED" in out.summary
    assert "do not state or assume their values" in out.summary
