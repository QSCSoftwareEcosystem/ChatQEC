"""Only module in this repo that imports lightstim.* — kept as a single seam so
upstream LightStim API churn (it's a young, fast-moving project) is isolated to
one file instead of leaking into the tool/server layer.

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


def _validate_domain(code_family: str, distance: int) -> None:
    """Pure, lightstim-independent validation of which (code_family, distance)
    combinations this tool actually supports.

    This is the single source of truth mirrored in
    `QecCircuitBuildInput.distance`'s Field description, and it is deliberately
    called before the `import lightstim` availability gate in `build_circuit`:
    "distance 7 isn't a valid bb_code preset" (or "color_code_666 needs an odd
    distance") is a fact about this tool's domain, not about lightstim, so it
    must be answerable even when the optional `lightstim` extra isn't
    installed.
    """
    if code_family not in _FAMILY_BUILDERS:
        raise ValueError(f"unknown code_family {code_family!r}")
    if code_family == "color_code_666" and distance % 2 == 0:
        raise ValueError(f"color_code_666 requires odd distance; got {distance}")
    if code_family == "bb_code" and distance not in _BB_CONFIGS:
        raise ValueError(
            f"bb_code only supports distance in {sorted(_BB_CONFIGS)}; got {distance}"
        )
    if code_family == "pqrm" and distance not in _PQRM_CONFIGS:
        raise ValueError(
            f"pqrm only supports distance in {sorted(_PQRM_CONFIGS)}; got {distance}"
        )
    # toric_code: any distance in this tool's schema range [_MIN_D, _MAX_D] is valid.


def _estimate_num_qubits(code_family: str, distance: int) -> int:
    """Cheap qubit-count probe used only to bound estimated work before the
    much more expensive full circuit assembly (MemoryExperiment.build() /
    CircuitBuilder). Constructs the real LightStim patch object only -- no
    rounds, no syndrome-extraction/detector assembly -- which measures well
    under 20ms even at this tool's largest supported distance/preset, against
    the pinned commit (see the `lightstim` extra in pyproject.toml).
    """
    if code_family == "toric_code":
        from lightstim.qec_code.surface_code.toric import ToricCode  # noqa: PLC0415

        return int(ToricCode(distance=distance).num_qubits)
    if code_family == "color_code_666":
        from lightstim.qec_code.color_code import ColorCode  # noqa: PLC0415

        return int(ColorCode(distance=distance, layout="superdense").num_qubits)
    if code_family == "bb_code":
        from lightstim.qec_code.BB_code import BBCode  # noqa: PLC0415

        cfg = _BB_CONFIGS[distance]
        return int(BBCode(l=cfg["l"], m=cfg["m"], A=cfg["A"], B=cfg["B"]).num_qubits)
    if code_family == "pqrm":
        from lightstim.qec_code.PQRM import PQRMPatch  # noqa: PLC0415

        cfg = _PQRM_CONFIGS[distance]
        patch = PQRMPatch(rx=cfg["rx"], rz=cfg["rz"], m=cfg["m"], punctured=True, shift=(0, 0))
        return int(patch.num_qubits)
    raise ValueError(f"unknown code_family {code_family!r}")  # unreachable; _validate_domain gates this


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
    # distance parity already checked by _validate_domain before this runs.
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
    # preset membership already checked by _validate_domain before this runs.
    cfg = _BB_CONFIGS[distance]

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
    # preset membership already checked by _validate_domain before this runs.
    cfg = _PQRM_CONFIGS[distance]

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

# Estimated qubit-count x rounds ceiling. Schema-valid input (distance<=15,
# rounds<=200) can otherwise measure ~3.6 minutes to build (toric_code d=15,
# r=200 -- ~9x over this ceiling) and, since qec_circuit_build's handler runs
# synchronously inside the MCP server's async call_tool, that blocks the whole
# server for the duration, not just this one call. Calibrated (see the fix-wave
# task report for the underlying measurements) so accepted combinations build
# in at most a handful of seconds; oversized *output* (independent of this
# ceiling -- circuit_text size tracks qubit count, not rounds, thanks to Stim's
# REPEAT-block compression) is bounded separately in qec_circuit_build.py.
_WORK_CEILING = 20_000


def build_circuit(
    *, code_family: str, distance: int, rounds: int, protocol: str = "memory"
) -> stim.Circuit:
    _require_memory_protocol(protocol)
    _validate_domain(code_family, distance)

    try:
        import lightstim  # noqa: F401, PLC0415
    except ImportError as e:
        raise LightStimUnavailable(f"lightstim not installed: {e}") from e

    num_qubits = _estimate_num_qubits(code_family, distance)
    estimated_work = num_qubits * rounds
    if estimated_work > _WORK_CEILING:
        raise ValueError(
            f"work ceiling exceeded: {code_family} at distance={distance} "
            f"(~{num_qubits} qubits) x rounds={rounds} = {estimated_work:,} estimated "
            f"qubit-rounds > {_WORK_CEILING:,}; reduce distance and/or rounds"
        )

    builder_fn = _FAMILY_BUILDERS[code_family]
    return builder_fn(distance, rounds)
