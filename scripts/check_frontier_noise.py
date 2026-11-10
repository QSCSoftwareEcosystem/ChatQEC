"""Appendix B check: does idle data-qubit noise explain the ChatQEC vs Claude count gap?

Runs decoder_compare (d=5 color code, p=0.002, seeds 42 and 43, 5000 shots each) on ChatQEC's circuit
and on the same circuit plus before_round_data_depolarization, and adds Claude's PyMatching variant
(decomposed error model, undecomposable mechanisms dropped). Uses the installed tools env, GPU 0:

    CUDA_VISIBLE_DEVICES=0 ~/.local/share/uv/tools/chatqec-mcp-tools/bin/python -I \
        scripts/check_frontier_noise.py

Output (2026-11-10): eval/compare/transcripts/claude_noise_check_20261110.json
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pymatching
import stim
from chatqec_mcp_tools.tools.decoder_compare import DecoderCompareInput, decoder_compare

logger = logging.getLogger(__name__)
P, SHOTS, SEEDS = 0.002, 5000, (42, 43)
DECODERS = ["tesseract", "cudaq_relay_bp", "pymatching"]


def _circuit(data_depol: bool) -> stim.Circuit:
    kw = {
        "distance": 5,
        "rounds": 5,
        "after_clifford_depolarization": P,
        "before_measure_flip_probability": P,
        "after_reset_flip_probability": P,
    }
    if data_depol:
        kw["before_round_data_depolarization"] = P
    return stim.Circuit.generated("color_code:memory_xyz", **kw)


def _pymatching_drop(c: stim.Circuit, seed: int) -> int:
    """Claude's PyMatching: decompose and drop mechanisms Stim cannot split into edges."""
    dem = c.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
    det, obs = c.compile_detector_sampler(seed=seed).sample(SHOTS, separate_observables=True)
    pred = pymatching.Matching.from_detector_error_model(dem).decode_batch(det)
    return int(np.any(pred != obs, axis=1).sum())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out: dict[str, dict[str, list[int]]] = {}
    for label, dd in (("chatqec_noise", False), ("plus_data_depol", True)):
        c = _circuit(dd)
        tot: dict[str, list[int]] = {}
        for seed in SEEDS:
            inp = DecoderCompareInput(circuit=str(c), shots=SHOTS, seed=seed, decoders=DECODERS)
            for r in decoder_compare(inp).results:
                if not r.ok:
                    logger.warning("%s seed %d %s failed: %s", label, seed, r.decoder, r.note)
                    continue
                e, s = tot.setdefault(r.decoder, [0, 0])
                tot[r.decoder] = [e + r.num_errors, s + r.shots]
            e, s = tot.setdefault("pymatching_drop", [0, 0])
            tot["pymatching_drop"] = [e + _pymatching_drop(c, seed), s + SHOTS]
        out[label] = tot
    logger.info(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
