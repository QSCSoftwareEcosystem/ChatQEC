"""decoder-bench adapter: items from the shipped gold, and a judge-free scorer."""

import json
from pathlib import Path

import pytest

from chatqec.api_types import Answer, ToolCall
from chatqec.eval.benchmarks import get_benchmark
from chatqec.eval.benchmarks.decoder_bench import DecoderBench, wilson

GOLD = (
    Path(__file__).resolve().parent.parent / "eval/benchmarks/decoder_bench/benchmark_results.csv"
)


@pytest.fixture(scope="module")
def bench() -> DecoderBench:
    return DecoderBench(gold_csv=GOLD)


def _item(bench, code, d, p, group):
    return next(
        i for i in bench.items()
        if i.meta["code"] == code and i.meta["d"] == d and i.meta["p"] == p
        and i.meta["group"] == group
    )  # fmt: skip


def _answer(args: dict, results: list[tuple[str, int, int]], text: str, best: str) -> Answer:
    out = {
        "ok": True,
        "best": best,
        "circuit_source": args.get("circuit_source", "chatqec"),
        "results": [
            {"decoder": d, "ok": True, "num_errors": k, "shots": n, "logical_error_rate": k / n}
            for d, k, n in results
        ],
    }
    return Answer(text=text, tool_calls=[ToolCall("decoder_compare", args, json.dumps(out))])


def test_registered_under_its_name():
    assert get_benchmark("decoder-bench") is DecoderBench


def test_items_skip_lattice_surgery_and_split_fast_and_slow_decoders(bench):
    items = bench.items()
    assert items and all(i.meta["code"] in {"surface", "color", "qldpc"} for i in items)
    fast = _item(bench, "surface", 7, 0.005, "fast")
    slow = _item(bench, "surface", 7, 0.005, "slow")
    assert set(fast.expected_args["decoders"]) == {"pymatching", "belief_find"}
    assert set(slow.expected_args["decoders"]) == {"relay_bp", "mwpf"}
    assert fast.expected_args["shots"] == 20_000 and slow.expected_args["shots"] == 4_000
    assert fast.expected_args["circuit_source"] == "decoder_bench"
    assert (
        "distance-7" in fast.prompt and "20,000 shots" in fast.prompt and "seed 42" in fast.prompt
    )


def test_only_configs_with_enough_expected_failures_are_scored_numerically(bench):
    item = _item(bench, "surface", 7, 0.0005, "fast")  # gold LERs ~1e-4 x 20k shots < 20
    assert not any(item.meta["resolvable"].values())
    item = _item(bench, "surface", 7, 0.005, "fast")
    assert all(item.meta["resolvable"].values())


def test_perfect_answer_scores_full_marks(bench):
    item = _item(bench, "surface", 7, 0.005, "fast")
    n = 20_000
    gold = item.gold
    results = [(d, round(gold[d] * n), n) for d in ("pymatching", "belief_find")]
    ans = _answer(dict(item.expected_args), results, "PyMatching is best.", "pymatching")
    s = bench.score(item, ans)
    assert s["routed"] and s["args_exact"] and s["numeric_in_ci"] == 1.0
    assert s["ranking_agreement"] == 1.0 and s["faithful_best"] and s["passed"]


def test_wrong_args_and_wrong_numbers_are_caught(bench):
    item = _item(bench, "surface", 7, 0.005, "fast")
    args = dict(item.expected_args, distance=5, circuit_source="chatqec")
    results = [("pymatching", 2000, 20_000), ("belief_find", 100, 20_000)]  # 10% and 0.5%
    s = bench.score(item, _answer(args, results, "BeliefFind is best.", "belief_find"))
    assert s["arg_match"]["distance"] is False and s["arg_match"]["circuit_source"] is False
    assert not s["args_exact"] and s["numeric_in_ci"] == 0.0 and not s["passed"]


def test_no_tool_call_scores_zero(bench):
    item = _item(bench, "surface", 7, 0.005, "fast")
    s = bench.score(item, Answer(text="PyMatching is usually best."))
    assert not s["routed"] and not s["passed"]


def test_wilson_interval_contains_rate_and_is_exact_at_zero():
    lo, hi = wilson(10, 1000)
    assert lo < 0.01 < hi
    assert wilson(0, 100)[0] == 0.0


def test_scorer_reads_python_repr_tool_output(bench):
    """chatQEC records tool output as a Python repr (single quotes), not JSON: the live
    smoke run scored 0 on numbers because the scorer only tried json.loads."""
    item = _item(bench, "surface", 7, 0.005, "fast")
    n = 20_000
    out = {
        "ok": True, "best": "pymatching", "circuit_source": "decoder_bench",
        "results": [
            {"decoder": d, "ok": True, "num_errors": round(item.gold[d] * n), "shots": n}
            for d in ("pymatching", "belief_find")
        ],
    }  # fmt: skip
    ans = Answer(
        text="PyMatching is best.",
        tool_calls=[ToolCall("decoder_compare", dict(item.expected_args), repr(out))],
    )
    s = bench.score(item, ans)
    assert s["numeric_in_ci"] == 1.0 and s["faithful_best"] and s["passed"]
