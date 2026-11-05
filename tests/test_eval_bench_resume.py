"""Resuming interrupted benchmark runs: what is missing, what errored, and when not to resume."""

import json
from pathlib import Path

from chatqec.eval.benchmarks.resume import (
    drift_reasons,
    is_being_written,
    plan_run,
    split_errors,
)


def _write_run(path: Path, rows: list[dict], settings: dict | None = None) -> None:
    header = {
        "model": "muse-glimmer",
        "chatqec": {"commit": "abc", "dirty": False},
        "tool_env": {"stim": "1.16.0", "ldpc": "2.4.1"},
        "settings": {
            "benchmark": "decoder-bench",
            "provider": "local",
            "mcp_server": "chatqec-mcp-tools",
            "tool_timeout_s": 1800.0,
            "system": "chatqec",
            "judge_model": None,
            **(settings or {}),
        },
    }
    path.write_text("\n".join(json.dumps(x) for x in [{"_header": header}, *rows]) + "\n")


def test_plan_counts_done_errored_and_missing(tmp_path):
    run = tmp_path / "decoder-bench_muse.jsonl"
    _write_run(run, [{"id": "a"}, {"id": "b", "error": "TimeoutError: MCP call exceeded 1800.0s"}])

    plan = plan_run(run, item_ids=["a", "b", "c", "d"])

    assert plan.benchmark == "decoder-bench" and plan.model == "muse-glimmer"
    assert plan.done == ["a"]
    assert plan.errored == ["b"]
    assert plan.missing == ["c", "d"]
    assert not plan.complete


def test_plan_is_complete_only_when_every_item_has_a_clean_row(tmp_path):
    run = tmp_path / "r.jsonl"
    _write_run(run, [{"id": "a"}, {"id": "b", "error": "boom"}])
    assert not plan_run(run, item_ids=["a", "b"]).complete
    _write_run(run, [{"id": "a"}, {"id": "b"}])
    assert plan_run(run, item_ids=["a", "b"]).complete


def test_plan_respects_the_runs_own_id_selection(tmp_path):
    """A run started with --ids covers only those items; others are not 'missing'."""
    run = tmp_path / "r.jsonl"
    _write_run(run, [{"id": "a"}], settings={"ids": "a,b"})
    assert plan_run(run, item_ids=["a", "b", "c"]).missing == ["b"]


def test_split_errors_moves_error_rows_aside_and_keeps_them(tmp_path):
    """Errored rows count as done for --resume; to retry them they must leave the run
    file (or the retry would add a duplicate row), but they are kept, not deleted."""
    run = tmp_path / "r.jsonl"
    _write_run(run, [{"id": "a"}, {"id": "b", "error": "timeout"}, {"id": "c"}])

    moved = split_errors(run)

    assert moved == 1
    kept = [json.loads(x) for x in run.read_text().splitlines()]
    assert "_header" in kept[0]
    assert [r["id"] for r in kept[1:]] == ["a", "c"]
    aside = run.with_name("r.errors.jsonl").read_text().splitlines()
    assert json.loads(aside[0])["id"] == "b"


def test_drift_flags_tool_env_and_code_changes():
    header = {"tool_env": {"stim": "1.16.0", "ldpc": "2.4.1"}}
    assert (
        drift_reasons(header, tool_env={"stim": "1.16.0", "ldpc": "2.4.1"}, ask_path_diff="") == []
    )
    reasons = drift_reasons(
        header, tool_env={"stim": "1.16.0", "ldpc": "2.5.0"}, ask_path_diff=" src/x.py | 2 +-"
    )
    assert any("ldpc 2.4.1 -> 2.5.0" in r for r in reasons)
    assert any("src/x.py" in r for r in reasons)


def test_a_file_another_process_is_writing_is_detected(tmp_path):
    run = tmp_path / "r.jsonl"
    cmdlines = [["python", "chatqec", "eval", "bench", "run", "x", "--out", str(run), "--resume"]]
    assert is_being_written(run, cmdlines)
    assert not is_being_written(tmp_path / "other.jsonl", cmdlines)
