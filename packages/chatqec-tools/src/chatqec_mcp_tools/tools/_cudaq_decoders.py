"""NVIDIA CUDA-Q QEC GPU decoders for decoder_compare (``[cudaq]`` extra).

Only this file imports ``cudaq_qec`` (upstream churn stays in one seam). Both
decoders are ``nv-qldpc-decoder`` configurations on the GPU:

  - ``cudaq_relay_bp``  RelayBP (bp_method=3, sequential relay, composition=1),
                        IBM's memory-strength BP variant for qLDPC codes.
  - ``cudaq_bposd``     BP+OSD with OSD-CS order 5.

API verified against the installed ``cudaq-qec-cu13==0.8.1`` with
``decoder_param_schema("nv-qldpc-decoder")`` / ``("srelay_bp")``; RelayBP
requires ``srelay_config`` and ``stopping_criterion`` in {All, NConv, FirstConv}.
Measured on a d=5, 5-round, p=0.002 color-code memory (2000 shots, RTX 6000 Ada):
RelayBP NConv(5) 0.007 in 0.27 s, All 0.005 in 1.07 s, FirstConv 0.009 in
0.15 s; BP+OSD-CS5 0.007 in 0.1 s. NConv is the default for speed vs accuracy.

The decoders predict an error vector ``e``; observable flips are ``O @ e mod 2``.
Needs an NVIDIA GPU: set CUDA_VISIBLE_DEVICES for the MCP server process to a
GPU with free memory (the local vLLM test servers occupy GPUs 0 and 1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import stim
    from numpy.typing import NDArray

CUDAQ_DECODERS = ("cudaq_relay_bp", "cudaq_bposd", "cudaq_sliding_window")

_BP_BATCH = 256
_RELAY = {
    "bp_method": 3,
    "composition": 1,
    "gamma0": 0.1,
    "gamma_dist": [-0.24, 0.66],
    "max_iterations": 80,
    "bp_seed": 42,
    "srelay_config": {
        "pre_iter": 80,
        "num_sets": 60,
        "stopping_criterion": "NConv",
        "stop_nconv": 5,
    },
}
_BPOSD = {"use_osd": True, "osd_method": 3, "osd_order": 5, "max_iterations": 30}


_DEFAULT_WINDOW = 5
_MIN_ROUND_COORDS = 3  # first, at least one full, and last round


def _round_size(dem: stim.DetectorErrorModel) -> int:
    """Detectors per full syndrome round, read from the DEM's time coordinate (last axis)."""
    from collections import Counter  # noqa: PLC0415

    coords = dem.get_detector_coordinates()
    per_round = Counter(c[-1] for c in coords.values() if c)
    if len(per_round) < _MIN_ROUND_COORDS:
        raise RuntimeError(
            "sliding window needs a multi-round memory circuit whose detectors carry a "
            "round (time) coordinate"
        )
    return max(per_round.values())


def cudaq_predict(
    name: str,
    dem: stim.DetectorErrorModel,
    det: NDArray[np.bool_],
    window_size: int | None = None,
    window_step: int | None = None,
) -> NDArray[np.bool_]:
    """Decode detector samples on the GPU and return predicted observable flips.

    Raises:
        ImportError: the ``cudaq`` extra is not installed.
        RuntimeError: no usable GPU or the decoder rejected its configuration.
    """
    import cudaq_qec as qec  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    qdem = qec.dem_from_stim_text(str(dem))
    h = np.asarray(qdem.detector_error_matrix, dtype=np.uint8)
    o = np.asarray(qdem.observables_flips_matrix, dtype=np.uint8)
    rates = np.asarray(qdem.error_rates)
    if name == "cudaq_sliding_window":
        # Validated on rotated surface-code memory (half-size first and last rounds): the
        # decoder needs time-ordered columns, then straddles only the final round.
        per_round = _round_size(dem)
        order = np.asarray(qec.get_sorted_pcm_column_indices(h, per_round))
        h, o, rates = h[:, order], o[:, order], rates[order]
    params = _RELAY if name == "cudaq_relay_bp" else _BPOSD
    try:
        if name == "cudaq_sliding_window":
            decoder = qec.get_decoder(
                "sliding_window",
                h,
                error_rate_vec=rates,
                window_size=window_size or _DEFAULT_WINDOW,
                step_size=window_step or 1,
                num_syndromes_per_round=per_round,
                num_boundary_syndromes=0,
                straddle_start_round=False,
                straddle_end_round=True,
                inner_decoder_name="nv-qldpc-decoder",
                inner_decoder_params={"use_sparsity": True, "bp_batch_size": _BP_BATCH, **_BPOSD},
            )
        else:
            decoder = qec.get_decoder(
                "nv-qldpc-decoder",
                h,
                error_rate_vec=rates,
                use_sparsity=True,
                bp_batch_size=_BP_BATCH,
                **params,
            )
        results = decoder.decode_batch(det.astype(np.float64))
    except ValueError as e:  # bad config or CUDA failure surfaces as ValueError too
        raise RuntimeError(str(e)) from e
    errors = np.array([np.round(r.result) for r in results], dtype=np.uint8)
    return ((errors @ o.T) % 2).astype(bool)
