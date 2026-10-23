"""add_uniform_noise must reproduce Stim's own generated-circuit noise exactly."""

import pytest
import stim

from chatqec_mcp_tools.tools._noise import add_uniform_noise


@pytest.mark.parametrize(
    "task",
    [
        "surface_code:rotated_memory_z",
        "surface_code:rotated_memory_x",
        "repetition_code:memory",
        "color_code:memory_xyz",
    ],
)
def test_matches_stim_generated_noise(task):
    p = 0.004
    ref = stim.Circuit.generated(
        task,
        distance=3,
        rounds=3,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )
    ours = add_uniform_noise(stim.Circuit.generated(task, distance=3, rounds=3), p)
    assert str(ours.detector_error_model()) == str(ref.detector_error_model())


def test_existing_noise_and_annotations_pass_through():
    c = stim.Circuit("X_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\nTICK")
    out = str(add_uniform_noise(c, 0.01))
    assert out.count("X_ERROR(0.1) 0") == 1 and "DETECTOR" in out and "TICK" in out
