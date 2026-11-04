"""FTPrimitiveBench adapter: items from generated gold, scored on circuit identity and numbers."""

import json

import pytest

from chatqec.api_types import Answer, ToolCall
from chatqec.eval.benchmarks import get_benchmark
from chatqec.eval.benchmarks.ft_primitive_bench import FtPrimitiveBench, reports_number

GOLD_ROWS = [
    {"id": "mem-x3z5-zbias10-p3e-3", "primitive": "memory", "x_distance": 3, "z_distance": 5,
     "noise": "pauli_biased", "p": 3e-3, "bias_axis": "Z", "bias_factor": 10.0,
     "gold_ler": 0.0125, "gold_shots": 200_000, "circuit_sha256": "abc"},
    {"id": "sg-d3-uniform-p3e-3", "primitive": "s_gate", "x_distance": 3, "z_distance": 3,
     "noise": "uniform", "p": 3e-3, "gold_ler": 0.0004, "gold_shots": 200_000,
     "circuit_sha256": "def"},
]  # fmt: skip


@pytest.fixture
def bench(tmp_path) -> FtPrimitiveBench:
    gold = tmp_path / "gold.jsonl"
    gold.write_text("".join(json.dumps(r) + "\n" for r in GOLD_ROWS))
    return FtPrimitiveBench(gold_jsonl=gold)


def _answer(item, *, sha, failures, shots=20_000, seed=42, text=None):
    rate = failures / shots
    out = {"ok": True, "circuit_sha256": sha, "num_failures": failures, "shots": shots,
           "logical_error_rate": rate}  # fmt: skip
    args = {"primitive": item.expected_args["primitive"], "shots": shots, "seed": seed}
    text = text if text is not None else f"The logical error rate is {rate:.4f}."
    return Answer(text=text, tool_calls=[ToolCall("ft_primitive_simulate", args, repr(out))])


def test_registered_and_items_carry_prompt_and_expected_circuit(bench):
    assert get_benchmark("ft-primitive-bench") is FtPrimitiveBench
    item = bench.items()[0]
    assert item.expected_tool == "ft_primitive_simulate"
    assert "X distance 3 and Z distance 5" in item.prompt
    assert "Z-biased" in item.prompt and "bias factor 10" in item.prompt
    assert "20,000 shots" in item.prompt and "seed 42" in item.prompt
    assert item.meta["circuit_sha256"] == "abc" and item.meta["resolvable"] is True


def test_low_rate_items_are_not_scored_numerically(bench):
    sg = bench.items()[1]
    assert "logical S gate" in sg.prompt and "distance-3" in sg.prompt
    assert sg.meta["resolvable"] is False  # 0.0004 x 20,000 = 8 expected failures


def test_correct_run_passes(bench):
    item = bench.items()[0]
    s = bench.score(item, _answer(item, sha="abc", failures=250))
    assert s["routed"] and s["circuit_match"] and s["numeric_in_ci"] == 1.0
    assert s["faithful"] and s["passed"]


def test_wrong_circuit_fails_even_with_good_numbers(bench):
    item = bench.items()[0]
    s = bench.score(item, _answer(item, sha="zzz", failures=250))
    assert not s["circuit_match"] and not s["passed"]


def test_number_not_reported_in_the_answer_is_unfaithful(bench):
    item = bench.items()[0]
    s = bench.score(item, _answer(item, sha="abc", failures=250, text="It is low."))
    assert not s["faithful"] and not s["passed"]


@pytest.mark.parametrize(
    "text, ok",
    [
        ("LER = 0.0125", True),
        ("logical error rate 1.25e-2", True),
        ("about 1.25%", True),
        ("1.25 \u00d7 10\u207b\u00b2", True),  # times-sign and superscript minus
        ("1.25 x 10^-2", True),
        ("LER = 0.02", False),
    ],
)
def test_reports_number_accepts_common_formats(text, ok):
    assert reports_number(text, 0.0125) is ok


def test_reports_zero_when_no_failures():
    assert reports_number("0 failures, LER 0.0", 0.0)
