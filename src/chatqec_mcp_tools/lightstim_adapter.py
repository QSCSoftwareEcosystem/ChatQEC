"""Only module in this repo that imports lightstim.* — see docs/planning
LightStim integration plan for why this seam exists (upstream churn isolation).

LightStim (https://github.com/QuTone/LightStim) is a higher-level framework built
on top of Stim that ships QEC code families Stim doesn't provide out of the box:
toric code, color code, bivariate-bicycle (BB) codes, and punctured quantum
Reed-Muller (PQRM) codes. This adapter is pinned to a specific LightStim commit
(see the `lightstim` extra in pyproject.toml) and calls into the *real* IR/
protocol API found at that commit (`lightstim/ir/*`, `lightstim/protocols/memory.py`,
`lightstim/qec_code/*`) — not a hypothetical shape.

Each supported family exposes a very different native construction API
(distance-parameterized geometry for toric/color code; discrete (l, m, A, B)
presets for BB codes; discrete (rx, rz, m) presets for PQRM). To keep
`build_circuit`'s signature uniform (`code_family, distance, rounds, protocol`),
this module maps `distance` onto the nearest/only matching native parameters per
family and raises `ValueError` with the supported values when there is no match.
"""

from __future__ import annotations

import stim

_MEMORY_PROTOCOL = "memory"

# BB codes are only defined for discrete (l, m, A, B) presets, not a continuous
# distance range. These are the standard presets from Bravyi et al. (arXiv:2308.07915)
# also used in LightStim's own notebooks/Memory/memory_BB.ipynb, restricted to the
# ones whose distance falls inside this tool's supported [3, 15] range.
_BB_CONFIGS: dict[int, dict[str, object]] = {
    6: {"l": 6, "m": 6, "A": [[3, 0], [0, 1], [0, 2]], "B": [[0, 3], [1, 0], [2, 0]]},
    10: {"l": 15, "m": 3, "A": [[9, 0], [0, 1], [0, 2]], "B": [[0, 0], [2, 0], [7, 0]]},
    12: {"l": 12, "m": 6, "A": [[3, 0], [0, 1], [0, 2]], "B": [[0, 3], [1, 0], [2, 0]]},
}

# PQRM is only defined for the (rx, rz, m) triples LightStim ships logical-operator
# presets for (see lightstim.qec_code.PQRM.pqrm_patch.LOG_PQRM_LEN_DICT). The dict
# values there are the logical-Z support length, which we use as the "distance" a
# caller of this tool asks for.
_PQRM_CONFIGS: dict[int, dict[str, object]] = {
    3: {"rx": 1, "rz": 2, "m": 4},
    7: {"rx": 1, "rz": 3, "m": 5},
}


class LightStimUnavailable(RuntimeError):
    pass


def _require_memory_protocol(protocol: str) -> None:
    if protocol != _MEMORY_PROTOCOL:
        raise ValueError(
            f"unsupported protocol {protocol!r}; only {_MEMORY_PROTOCOL!r} is implemented"
        )


def _build_toric(distance: int, rounds: int) -> stim.Circuit:
    from lightstim.protocols.memory import MemoryExperiment  # noqa: PLC0415
    from lightstim.qec_code.surface_code.toric import (  # noqa: PLC0415
        ToricCode,
        ToricCodeExtractionBlock,
    )

    patch = ToricCode(distance=distance)
    experiment = MemoryExperiment(
        qec_patch=patch,
        extraction_block_class=ToricCodeExtractionBlock,
        rounds=rounds,
        basis="Z",
    )
    return experiment.build()


def _build_color_code_666(distance: int, rounds: int) -> stim.Circuit:
    if distance % 2 == 0:
        raise ValueError(f"color_code_666 requires odd distance; got {distance}")

    from lightstim.protocols.memory import MemoryExperiment  # noqa: PLC0415
    from lightstim.qec_code.color_code import (  # noqa: PLC0415
        ColorCode,
        ColorCodeExtractionBlock,
    )

    patch = ColorCode(distance=distance, layout="superdense")
    experiment = MemoryExperiment(
        qec_patch=patch,
        extraction_block_class=ColorCodeExtractionBlock,
        rounds=rounds,
        basis="Z",
    )
    return experiment.build()


def _build_bb_code(distance: int, rounds: int) -> stim.Circuit:
    cfg = _BB_CONFIGS.get(distance)
    if cfg is None:
        raise ValueError(
            f"bb_code only supports distance in {sorted(_BB_CONFIGS)}; got {distance}"
        )

    from lightstim.protocols.memory import MemoryExperiment  # noqa: PLC0415
    from lightstim.qec_code.BB_code import BBCode, BBCodeExtractionBlock  # noqa: PLC0415

    patch = BBCode(l=cfg["l"], m=cfg["m"], A=cfg["A"], B=cfg["B"])
    experiment = MemoryExperiment(
        qec_patch=patch,
        extraction_block_class=BBCodeExtractionBlock,
        rounds=rounds,
        basis="Z",
    )
    return experiment.build()


def _build_pqrm(distance: int, rounds: int) -> stim.Circuit:
    cfg = _PQRM_CONFIGS.get(distance)
    if cfg is None:
        raise ValueError(
            f"pqrm only supports distance in {sorted(_PQRM_CONFIGS)}; got {distance}"
        )

    from lightstim.ir.builder import CircuitBuilder  # noqa: PLC0415
    from lightstim.ir.qec_system import QECSystem  # noqa: PLC0415
    from lightstim.ir.tracker import SyndromeTracker  # noqa: PLC0415
    from lightstim.qec_code.PQRM import (  # noqa: PLC0415
        PQRMExtractionBlock,
        PQRMLogicalOpSet,
        PQRMPatch,
    )

    # PQRM needs an explicit hypercube-encoding step before syndrome extraction
    # (see notebooks/Memory/memory_PQRM.ipynb) — it cannot go through the generic
    # MemoryExperiment(basis=...) reset-based initialization used by the other
    # three families.
    patch = PQRMPatch(rx=cfg["rx"], rz=cfg["rz"], m=cfg["m"], punctured=True, shift=(0, 0))
    system = QECSystem()
    system.add_patch(patch, name="pqrm")
    tracker = SyndromeTracker(
        num_qubits=system.num_qubits, expected_num_logicals=system.num_logicals
    )
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    builder.write_coordinates()
    PQRMLogicalOpSet().hypercube_encode(builder, patch, target_state="Z", patch_name="pqrm")
    builder.stabilizer_canonicalization()
    se_block = PQRMExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se_block.circuit, rounds=rounds)
    data_indices = sorted(system.data_indices)
    builder.apply_data_readout(final_measurements=dict.fromkeys(data_indices, "Z"))
    return builder.circuit


_FAMILY_BUILDERS = {
    "toric_code": _build_toric,
    "color_code_666": _build_color_code_666,
    "bb_code": _build_bb_code,
    "pqrm": _build_pqrm,
}


def build_circuit(
    *, code_family: str, distance: int, rounds: int, protocol: str = "memory"
) -> stim.Circuit:
    try:
        import lightstim  # noqa: F401, PLC0415
    except ImportError as e:
        raise LightStimUnavailable(f"lightstim not installed: {e}") from e

    _require_memory_protocol(protocol)

    builder_fn = _FAMILY_BUILDERS.get(code_family)
    if builder_fn is None:
        raise ValueError(f"unknown code_family {code_family!r}")

    return builder_fn(distance, rounds)
