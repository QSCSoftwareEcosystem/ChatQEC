"""Audit decoder-bench's shipped gold for the (config, decoder) pairs a run failed on.

decoder-bench ships logical error rates in ``results/benchmark_results.csv``. For some pairs
chatQEC's decoder_compare (which rebuilds decoder-bench's circuits; see its
``circuit_source='decoder_bench'``) lands outside the gold. This script asks whether the gold
or the tool is off, using decoder-bench's OWN code at its pinned commit:

1. read run files, find each resolvable decoder whose gold lies outside the 95% Wilson
   interval of the count chatQEC's tool measured;
2. rebuild the circuit with decoder-bench's ``generator.gen_*`` and check its SHA-256 against
   the ``circuit_sha256`` the tool reported (same circuit, or the comparison is void);
3. sample it with Stim and decode with decoder-bench's registered decoder class and its
   default parameters, at ``--shot-factor`` times the item's shots and a seed different
   from the item's;
4. write one JSON line per pair: native count, Wilson interval, whether the shipped gold is
   consistent with it, and a two-proportion z-test between native and chatQEC's count.

Run it in decoder-bench's native env (its own pinned requirements plus relay-bp and mwpf;
recipe and build commands in eval/benchmarks/decoder_bench/native-requirements.txt):

    data/benchmarks/decoder-bench-native/bin/python \\
        scripts/benchmarks/decoder_bench_gold_audit.py \\
        --decoder-bench data/benchmarks/decoder-bench \\
        --out eval/benchmarks/decoder_bench/gold_audit.jsonl \\
        eval/results/decoder-bench_*.jsonl
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

PINNED_COMMIT = "9733a55a3a0e37c618294bd2ebbe0a8d4c334943"
AUDIT_SEED = 4242  # never the item seed (42)
Z95 = 1.96
# chatQEC decoder name -> decoder-bench registry name
NATIVE_NAME = {
    "belief_find": "belieffind",
    "bplsd": "bplsd",
    "pymatching": "pymatching",
    "relay_bp": "relay-bp",
    "mwpf": "mwpf",
}


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> float:
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    return 0.0 if se == 0 else (k1 / n1 - k2 / n2) / se


def _tool_output(call: dict[str, Any]) -> dict[str, Any]:
    out = call["output"]
    return ast.literal_eval(out) if isinstance(out, str) else out


def failing_pairs(run_files: list[Path]) -> list[dict[str, Any]]:
    """Pairs whose shipped gold is outside chatQEC's measured 95% interval, de-duplicated."""
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for path in run_files:
        lines = path.read_text().splitlines()
        for line in lines[1:]:
            row = json.loads(line)
            if row.get("error"):
                continue
            meta, gold = row["meta"], row["gold"]
            for call in row["tool_calls"]:
                if call["name"] != "decoder_compare":
                    continue
                out = _tool_output(call)
                for res in out.get("results", []):
                    name = res.get("decoder")
                    if not res.get("ok") or not meta["resolvable"].get(name):
                        continue
                    k, n = int(res["num_errors"]), int(res["shots"])
                    lo, hi = wilson(k, n)
                    if lo <= gold[name] <= hi:
                        continue
                    key = (meta["code"], meta["d"], meta["p"], name)
                    seen.setdefault(
                        key,
                        {
                            "code": meta["code"],
                            "d": meta["d"],
                            "p": meta["p"],
                            "decoder": name,
                            "gold_csv": gold[name],
                            "chatqec_errors": k,
                            "chatqec_shots": n,
                            "chatqec_circuit_sha256": out.get("circuit_sha256"),
                            "seen_in": [],
                        },
                    )["seen_in"].append(f"{path.name}:{row['id']}")
    return list(seen.values())


def _native_circuit(gen: Any, code: str, d: int, p: float) -> Any:
    if code == "surface":
        return gen.gen_surface_circuit((d, p, "z"))
    if code == "color":
        return gen.gen_color_circuit((d, p))
    if code == "qldpc":
        return gen.gen_qldpc_circuit((d, p, "z"))
    raise ValueError(f"no decoder-bench generator for code {code!r}")


def native_rate(bench_pkg: Path, pair: dict[str, Any], shots: int) -> dict[str, Any]:
    """Count logical failures with decoder-bench's own circuit, sampler and decoder class."""
    sys.path.insert(0, str(bench_pkg))
    import decoders  # noqa: F401  (registers the decoder classes)
    import generator as gen
    from common.build_circuit import dem_to_check_matrices
    from sampler import DecoderRegistry, DecoderState

    circuit = _native_circuit(gen, pair["code"], pair["d"], pair["p"])
    sha = hashlib.sha256(str(circuit).encode()).hexdigest()
    dem = circuit.detector_error_model()
    check, obs, priors = dem_to_check_matrices(dem)
    state = DecoderState(
        check_matrix=check, obs_matrix=obs, priors=priors, circuit=circuit, dem=dem
    )
    decoder = DecoderRegistry.get(NATIVE_NAME[pair["decoder"]])(state)
    dets, observables = circuit.compile_detector_sampler(seed=AUDIT_SEED).sample(
        shots, separate_observables=True
    )
    t0 = time.time()
    preds = decoder.decode_batch(dets.astype(np.uint8))
    ok = decoder.verify_batch(preds, observables.astype(np.uint8))
    errors = int(shots - sum(bool(v) for v in ok))
    return {
        "native_circuit_sha256": sha,
        "native_errors": errors,
        "native_shots": shots,
        "native_seed": AUDIT_SEED,
        "native_decode_seconds": round(time.time() - t0, 1),
    }


def audit(pair: dict[str, Any], native: dict[str, Any]) -> dict[str, Any]:
    k, n = native["native_errors"], native["native_shots"]
    lo, hi = wilson(k, n)
    z = two_proportion_z(pair["chatqec_errors"], pair["chatqec_shots"], k, n)
    return {
        **pair,
        **native,
        "native_rate": k / n,
        "native_ci95": [lo, hi],
        "same_circuit": native["native_circuit_sha256"] == pair["chatqec_circuit_sha256"],
        "csv_gold_consistent_with_native": lo <= pair["gold_csv"] <= hi,
        "z_chatqec_vs_native": round(z, 3),
        "chatqec_consistent_with_native": abs(z) < Z95,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--decoder-bench", type=Path, required=True, help="pinned checkout")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shot-factor", type=int, default=5)
    args = ap.parse_args()

    head = subprocess.run(
        ["git", "-C", str(args.decoder_bench), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != PINNED_COMMIT:
        raise SystemExit(f"decoder-bench checkout is at {head}, expected {PINNED_COMMIT}")

    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["code"], r["d"], r["p"], r["decoder"]))
    pairs = failing_pairs(args.runs)
    logger.info("%d failing pair(s), %d already audited", len(pairs), len(done))
    with args.out.open("a") as f:
        for pair in pairs:
            key = (pair["code"], pair["d"], pair["p"], pair["decoder"])
            if key in done:
                continue
            shots = pair["chatqec_shots"] * args.shot_factor
            row = audit(pair, native_rate(args.decoder_bench / "decoder_bench", pair, shots))
            f.write(json.dumps(row) + "\n")
            f.flush()
            logger.info(
                "%s d=%s p=%s %s: native %d/%d (CSV gold %.3g consistent=%s; chatqec z=%.2f) same_circuit=%s",
                *key,
                row["native_errors"],
                row["native_shots"],
                row["gold_csv"],
                row["csv_gold_consistent_with_native"],
                row["z_chatqec_vs_native"],
                row["same_circuit"],
            )


if __name__ == "__main__":
    main()
