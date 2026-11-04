"""Benchmark run loop: one JSONL row per item, a provenance header, resume, and a summary."""

import json
from typing import Any

from chatqec.api_types import Answer, ToolCall
from chatqec.eval.benchmarks.base import Benchmark, BenchItem
from chatqec.eval.benchmarks.run import run_benchmark, summarize


class _Toy(Benchmark):
    name = "toy"

    def items(self) -> list[BenchItem]:
        return [
            BenchItem(id=f"toy/{i}", prompt=f"q{i}", expected_tool="t", expected_args={}, gold={})
            for i in range(3)
        ]

    def score(self, item: BenchItem, answer: Answer) -> dict[str, Any]:
        return {"routed": bool(answer.tool_calls), "passed": item.id != "toy/1"}


def _ask(prompt: str) -> Answer:
    return Answer(text=f"answer to {prompt}", tool_calls=[ToolCall("t", {"x": 1}, '{"ok": true}')])


def test_writes_header_then_one_row_per_item(tmp_path):
    out = tmp_path / "run.jsonl"
    run_benchmark(_Toy(), _ask, out, provenance={"model": "m"})
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert lines[0]["_header"]["model"] == "m"
    rows = lines[1:]
    assert [r["id"] for r in rows] == ["toy/0", "toy/1", "toy/2"]
    assert rows[0]["answer_text"] == "answer to q0"  # verbatim, for quoting in the paper
    assert rows[0]["tool_calls"] == [{"name": "t", "input": {"x": 1}, "output": '{"ok": true}'}]
    assert rows[1]["score"]["passed"] is False and "seconds" in rows[0]


def test_resume_skips_items_already_in_the_file(tmp_path):
    out = tmp_path / "run.jsonl"
    asked: list[str] = []
    run_benchmark(_Toy(), _ask, out, provenance={}, limit=2)

    def counting(prompt: str) -> Answer:
        asked.append(prompt)
        return _ask(prompt)

    run_benchmark(_Toy(), counting, out, provenance={}, resume=True)
    assert asked == ["q2"]
    assert sum(1 for line in out.read_text().splitlines() if '"_header"' not in line) == 3


def test_an_exception_is_recorded_as_a_failed_row_not_raised(tmp_path):
    def broken(prompt: str) -> Answer:
        raise RuntimeError("model down")

    out = tmp_path / "run.jsonl"
    run_benchmark(_Toy(), broken, out, provenance={}, limit=1)
    row = json.loads(out.read_text().splitlines()[1])
    assert row["error"] == "RuntimeError: model down" and row["score"]["passed"] is False


def test_summary_averages_only_applicable_scores(tmp_path):
    rows = [
        {"score": {"passed": True, "routed": True, "numeric_in_ci": 1.0}},
        {"score": {"passed": False, "routed": True, "numeric_in_ci": None}},
        {"score": {"passed": False, "routed": False}},
    ]
    s = summarize(rows)
    assert s["n"] == 3 and abs(s["passed"] - 1 / 3) < 1e-9
    assert abs(s["routed"] - 2 / 3) < 1e-9
    assert s["numeric_in_ci"] == 1.0 and s["numeric_in_ci_n"] == 1
