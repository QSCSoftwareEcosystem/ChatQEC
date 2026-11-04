"""decoder-bench adapter: decoder comparisons on its SI1000 memory circuits, scored on its gold.

Gold is decoder-bench's shipped ``results/benchmark_results.csv`` (0.5M to 10M shots per
row, commit 9733a55), copied to ``eval/benchmarks/decoder_bench/``. Lattice-surgery rows are
skipped: chatQEC has no lattice-surgery decoder comparison. Each memory configuration yields
two items, because the decoders differ ~1000x in speed: a fast group (PyMatching, BeliefFind,
BP+LSD) at 20,000 shots and a slow group (IBM Relay-BP, MWPF) at 4,000 shots, both seed 42.

A decoder is scored numerically only when the gold rate times the item's shots gives at least
``MIN_EXPECTED_FAILURES`` failures; below that a 95% interval says little. The other scores
(routing, argument extraction, ranking, faithfulness) apply to every item.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any

from chatqec.api_types import Answer
from chatqec.eval.benchmarks import register_benchmark
from chatqec.eval.benchmarks.base import BenchItem, Benchmark, last_call, tool_output

MIN_EXPECTED_FAILURES = 20
SEPARABLE_RATIO = 1.5  # rank only decoder pairs whose gold rates differ at least this much
SEED = 42
_Z95 = 1.959963984540054
_FILE = re.compile(r"(surface|color|qldpc)_circuit_(\d+)_([0-9]*\.?[0-9]+)(?:_[xz])?\.h5$")
# decoder-bench name -> (chatQEC decoder name, name used in the prompt, aliases in answers)
_DECODERS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "pymatching": ("pymatching", "PyMatching", ("pymatching", "mwpm")),
    "belieffind": (
        "belief_find",
        "BeliefFind (BP + union-find)",
        ("belieffind", "belief_find", "belief-find", "belief find", "union-find", "bp+uf"),
    ),
    "bplsd": ("bplsd", "BP+LSD", ("bplsd", "bp+lsd", "bp-lsd", "bp_lsd")),
    "relay-bp": ("relay_bp", "IBM Relay-BP", ("relay-bp", "relay_bp", "relaybp", "relay bp")),
    "mwpf": ("mwpf", "MWPF", ("mwpf", "parity factor")),
}
_GROUPS = {
    "fast": (("pymatching", "belieffind", "bplsd"), 20_000),
    "slow": (("relay-bp", "mwpf"), 4_000),
}
_CODES = {
    "surface": ("surface_code_rotated", "rotated surface code"),
    "color": ("color_code", "color code"),
    "qldpc": ("bb_code", "bivariate bicycle (BB) qLDPC code"),
}


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval, exact at 0 and n."""
    p = k / n
    den = 1 + _Z95**2 / n
    centre = (p + _Z95**2 / (2 * n)) / den
    half = _Z95 * math.sqrt(p * (1 - p) / n + _Z95**2 / (4 * n * n)) / den
    return (0.0 if k == 0 else max(0.0, centre - half), 1.0 if k == n else min(1.0, centre + half))


def _load_gold(path: Path) -> dict[tuple[str, int, float], dict[str, float]]:
    gold: dict[tuple[str, int, float], dict[str, float]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            m = _FILE.match(row["filename"])
            if not m or row["decoder"] not in _DECODERS:
                continue
            key = (m.group(1), int(m.group(2)), float(m.group(3)))
            gold.setdefault(key, {})[row["decoder"]] = float(row["logical_error_rate"])
    return gold


def _prompt(code: str, d: int, p: float, names: list[str], shots: int) -> str:
    listed = ", ".join(names[:-1]) + f" and {names[-1]}" if len(names) > 1 else names[0]
    return (
        f"Using decoder-bench's circuit (SI1000 circuit-level noise), compare {listed} on a "
        f"distance-{d} {_CODES[code][1]} memory experiment at p = {p}, with {shots:,} shots and "
        f"seed {SEED}. Rank the decoders by logical error rate."
    )


@register_benchmark("decoder-bench")
class DecoderBench(Benchmark):
    def __init__(self, gold_csv: Path | str) -> None:
        self._gold = _load_gold(Path(gold_csv))

    def items(self) -> list[BenchItem]:
        out = []
        for (code, d, p), rates in sorted(self._gold.items()):
            for group, (members, shots) in _GROUPS.items():
                present = [m for m in members if m in rates]
                if not present:
                    continue
                ours = [_DECODERS[m][0] for m in present]
                gold = {_DECODERS[m][0]: rates[m] for m in present}
                out.append(
                    BenchItem(
                        id=f"decoder-bench/{code}-d{d}-p{p}-{group}",
                        prompt=_prompt(code, d, p, [_DECODERS[m][1] for m in present], shots),
                        expected_tool="decoder_compare",
                        expected_args={
                            "circuit_source": "decoder_bench",
                            "code_family": _CODES[code][0],
                            "distance": d,
                            "p": p,
                            "decoders": ours,
                            "shots": shots,
                            "seed": SEED,
                        },
                        gold=gold,
                        meta={
                            "code": code,
                            "d": d,
                            "p": p,
                            "group": group,
                            "resolvable": {
                                k: v * shots >= MIN_EXPECTED_FAILURES for k, v in gold.items()
                            },
                        },
                    )
                )
        return out

    def score(self, item: BenchItem, answer: Answer) -> dict[str, Any]:
        call = last_call(answer, "decoder_compare", circuit_source="decoder_bench")
        if call is None:
            return {
                "routed": False,
                "passed": False,
                "tool_calls": [c.name for c in answer.tool_calls],
            }
        args, out = call.input, tool_output(call)
        arg_match = {k: _same(k, args.get(k), v) for k, v in item.expected_args.items()}
        measured = {
            r["decoder"]: (int(r["num_errors"]), int(r["shots"]))
            for r in out.get("results", [])
            if r.get("ok") and r.get("shots")
        }
        resolvable = [k for k, ok in item.meta["resolvable"].items() if ok]
        in_ci = [
            k in measured and wilson(*measured[k])[0] <= item.gold[k] <= wilson(*measured[k])[1]
            for k in resolvable
        ]
        numeric = sum(in_ci) / len(in_ci) if in_ci else None
        ranking = _ranking_agreement(item.gold, measured, resolvable)
        best = str(out.get("best", ""))
        faithful = bool(best) and _mentions(answer.text, best)
        tool_ok = bool(out.get("ok"))
        args_exact = all(arg_match.values())
        return {
            "routed": True,
            "tool_ok": tool_ok,
            "arg_match": arg_match,
            "args_exact": args_exact,
            "numeric_in_ci": numeric,
            "ranking_agreement": ranking,
            "faithful_best": faithful,
            "passed": tool_ok and args_exact and numeric in (1.0, None) and ranking in (1.0, None),
        }


def _same(key: str, got: Any, want: Any) -> bool:
    if got is None:
        return False
    if key == "decoders":
        return set(got) == set(want)
    if key == "p":
        return isinstance(got, int | float) and math.isclose(float(got), want, rel_tol=1e-9)
    return bool(got == want)


def _ranking_agreement(
    gold: dict[str, float], measured: dict[str, tuple[int, int]], names: list[str]
) -> float | None:
    """Share of clearly separated gold pairs whose order the measured rates reproduce."""
    pairs = [
        (a, b)
        for i, a in enumerate(names)
        for b in names[i + 1 :]
        if min(gold[a], gold[b]) > 0
        and max(gold[a], gold[b]) / min(gold[a], gold[b]) >= SEPARABLE_RATIO
    ]
    if not pairs:
        return None
    agree = 0
    for a, b in pairs:
        if a in measured and b in measured:
            ra, rb = measured[a][0] / measured[a][1], measured[b][0] / measured[b][1]
            agree += (ra < rb) == (gold[a] < gold[b])
    return agree / len(pairs)


def _mentions(text: str, decoder: str) -> bool:
    low = text.lower()
    aliases = next((v[2] for v in _DECODERS.values() if v[0] == decoder), (decoder,))
    return any(a in low for a in aliases)
