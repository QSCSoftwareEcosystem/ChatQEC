"""FTPrimitiveBench adapter: surface-code primitives under structured noise, scored on its gold.

Gold comes from ``scripts/benchmarks/ft_primitive_gold.py``, which runs the FTPrimitiveBench
library (commit b75fbd1) directly at 200,000 shots. Every item asks for 20,000 shots and seed
42. The decisive check is circuit identity: the tool reports the noisy circuit's sha256, and it
must equal the gold circuit's. That one comparison covers everything the model had to extract
from the prompt (primitive, both distances, every round count, noise family and parameters).

A rate is scored numerically only when gold LER x 20,000 gives at least MIN_EXPECTED_FAILURES
failures; the answer must also state the rate the tool returned (``reports_number``).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from chatqec.api_types import Answer
from chatqec.eval.benchmarks import register_benchmark
from chatqec.eval.benchmarks.base import BenchItem, Benchmark, last_call, tool_output
from chatqec.eval.benchmarks.decoder_bench import MIN_EXPECTED_FAILURES, wilson

SHOTS = 20_000
SEED = 42
_REL_TOL = 0.02  # a reported number counts if within 2% of the tool's rate (rounding)
_NUM = re.compile(
    r"(?P<m>\d+(?:\.\d+)?)\s*(?:e(?P<e>[-+]?\d+)|[x\u00d7]\s*10\s*\^?\s*\{?(?P<p>[-\u2212\u207b]?\s*\d+)\}?)?\s*(?P<pct>%)?",
    re.IGNORECASE,
)
# superscript digits and minus signs (U+207B, U+2212) that models write in "10⁻²"
_SUP = str.maketrans(
    "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b\u2212", "0123456789--"
)


def _noise_phrase(row: dict[str, Any]) -> str:
    p = row["p"]
    if row["noise"] == "pauli_biased":
        return f"{row['bias_axis']}-biased Pauli noise with p = {p} and bias factor {row['bias_factor']:g}"
    if row["noise"] == "measurement_biased":
        return f"measurement-biased noise with p = {p} and measurement bias factor {row['bias_factor']:g}"
    if row["noise"] == "nonuniform":
        return f"non-uniform noise with p = {p} and relative scatter sigma = {row['sigma']:g}"
    return f"uniform depolarizing circuit noise with p = {p}"


def _patch(row: dict[str, Any]) -> str:
    xd, zd = row["x_distance"], row["z_distance"]
    if xd == zd:
        return f"distance-{xd} rotated surface-code patch"
    return f"rotated surface-code patch with X distance {xd} and Z distance {zd}"


def _prompt(row: dict[str, Any]) -> str:
    d = max(row["x_distance"], row["z_distance"])
    what = {
        "memory": f"a logical memory experiment on a {_patch(row)} ({d} rounds, Z-basis readout)",
        "transversal_h": (
            f"a transversal Hadamard on a {_patch(row)}: {d} rounds before and {d} rounds "
            "after one transversal H layer, Z-basis readout"
        ),
        "lattice_surgery": (
            f"a ZZ lattice-surgery merge and split between two patches, each a {_patch(row)}, "
            f"with bridge length 1 and {d} rounds before, during and after the merge"
        ),
        "s_gate": (
            f"a logical S gate (ZZ lattice surgery with a teleported Y magic state) on "
            f"distance-{d} patches, with {d} rounds before, during and after the merge and one "
            "boundary round"
        ),
    }[row["primitive"]]
    return (
        f"Using FTPrimitiveBench, simulate {what}, under {_noise_phrase(row)}. Use {SHOTS:,} "
        f"shots and seed {SEED}, decode with PyMatching, and report the logical error rate with "
        "its 95% interval."
    )


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUM.finditer(text.translate(_SUP)):
        value = float(m.group("m"))
        if m.group("e"):
            value *= 10 ** int(m.group("e"))
        elif m.group("p"):
            value *= 10 ** int(m.group("p").replace(" ", ""))
        if m.group("pct"):
            value /= 100
        out.append(value)
    return out


def reports_number(text: str, value: float) -> bool:
    """Whether ``text`` states ``value`` (decimal, e-notation, x10^, superscripts or percent)."""
    if value == 0:
        return any(n == 0 for n in _numbers(text))
    return any(math.isclose(n, value, rel_tol=_REL_TOL) for n in _numbers(text))


@register_benchmark("ft-primitive-bench")
class FtPrimitiveBench(Benchmark):
    def __init__(self, gold_jsonl: Path | str) -> None:
        self._rows = [
            json.loads(line) for line in Path(gold_jsonl).read_text().splitlines() if line
        ]

    def items(self) -> list[BenchItem]:
        return [
            BenchItem(
                id=f"ft-primitive-bench/{row['id']}",
                prompt=_prompt(row),
                expected_tool="ft_primitive_simulate",
                expected_args={"primitive": row["primitive"], "shots": SHOTS, "seed": SEED},
                gold={"logical_error_rate": row["gold_ler"]},
                meta={
                    "primitive": row["primitive"],
                    "noise": row["noise"],
                    "circuit_sha256": row["circuit_sha256"],
                    "resolvable": row["gold_ler"] * SHOTS >= MIN_EXPECTED_FAILURES,
                },
            )
            for row in self._rows
        ]

    def score(self, item: BenchItem, answer: Answer) -> dict[str, Any]:
        call = last_call(answer, "ft_primitive_simulate")
        if call is None:
            return {
                "routed": False,
                "passed": False,
                "tool_calls": [c.name for c in answer.tool_calls],
            }
        out = tool_output(call)
        circuit_match = out.get("circuit_sha256") == item.meta["circuit_sha256"]
        shots, failures = int(out.get("shots") or 0), int(out.get("num_failures") or 0)
        seed_ok = call.input.get("seed") == SEED and shots == SHOTS
        numeric = None
        if item.meta["resolvable"]:
            lo, hi = wilson(failures, shots) if shots else (1.0, 0.0)
            numeric = 1.0 if lo <= item.gold["logical_error_rate"] <= hi else 0.0
        rate = out.get("logical_error_rate")
        faithful = isinstance(rate, int | float) and reports_number(answer.text, float(rate))
        return {
            "routed": True,
            "tool_ok": bool(out.get("ok")),
            "circuit_match": circuit_match,
            "args_exact": circuit_match and seed_ok,
            "numeric_in_ci": numeric,
            "faithful_best": faithful,
            "faithful": faithful,
            "passed": bool(out.get("ok"))
            and circuit_match
            and seed_ok
            and numeric in (1.0, None)
            and faithful,
        }
