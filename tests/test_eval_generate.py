"""Generation pipeline tests: seed loading + generate_candidates."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from chatqec.api import Answer, ToolCall
from chatqec.eval.generate import SeedError, generate_candidates, load_seed_cases


def _write_seeds(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "seeds.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_load_seed_cases_round_trips(tmp_path: Path):
    p = _write_seeds(
        tmp_path,
        [
            {
                "id": "route-stim-001",
                "subtype": "code_execution",
                "question": "Simulate H 0\nM 0 for 100 shots.",
                "complexity": "simple",
                "expected_tools": ["stim_simulate"],
                "target_capability": ["stim"],
                "is_adversarial": False,
            }
        ],
    )
    seeds = load_seed_cases(p)
    assert len(seeds) == 1
    assert seeds[0].expected_tools == ["stim_simulate"]
    assert seeds[0].expected_numeric_facts == {}


def test_load_seed_cases_rejects_ibm_submit_execution(tmp_path: Path):
    p = _write_seeds(
        tmp_path,
        [
            {
                "id": "bad-001",
                "subtype": "hardware_mapping",
                "question": "Run this on real hardware.",
                "expected_tools": ["ibm_submit_execution"],
            }
        ],
    )
    with pytest.raises(SeedError):
        load_seed_cases(p)


def _judge_client_returning(payload: dict) -> MagicMock:
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text=json.dumps(payload))]
    client.messages.create.return_value = msg
    return client


def test_generate_candidates_writes_log_and_returns_candidate(tmp_path: Path):
    p = _write_seeds(
        tmp_path,
        [
            {
                "id": "route-stim-001",
                "subtype": "code_execution",
                "question": "Simulate H 0\nM 0 for 100 shots.",
                "complexity": "simple",
                "expected_tools": ["stim_simulate"],
                "target_capability": ["stim"],
            }
        ],
    )
    seeds = load_seed_cases(p)

    bot = MagicMock()
    bot.ask.return_value = Answer(
        text="Roughly an even split of 0s and 1s.",
        tool_calls=[ToolCall(name="stim_simulate", input={"shots": 100}, output="{}")],
        confidence="medium",
    )
    judge_client = _judge_client_returning(
        {
            "verdict": "correct",
            "verdict_reasoning": "Matches expected Bell-like output.",
            "gold_answer_summary": "~50/50 split of 0s and 1s.",
            "routing_flag": "match",
            "routing_flag_reasoning": "stim_simulate was invoked.",
        }
    )
    log_path = tmp_path / "logs" / "batch.jsonl"

    candidates = generate_candidates(
        seeds=seeds,
        bot=bot,
        judge_client=judge_client,
        judge_model="claude-haiku-4-5",
        log_path=log_path,
    )

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand["id"] == "route-stim-001"
    assert cand["gold_answer_summary"] == "~50/50 split of 0s and 1s."
    assert cand["_judge_verdict"] == "correct"
    assert cand["_routing_flag"] == "match"

    log_lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == 1
    record = json.loads(log_lines[0])
    assert record["tool_calls"] == [
        {"name": "stim_simulate", "input": {"shots": 100}, "output": "{}"}
    ]
    assert record["judge_verdict"] == "correct"
    assert "elapsed_seconds" in record


def test_generate_candidates_skips_and_logs_unsafe_tool_without_calling_judge(tmp_path: Path):
    p = _write_seeds(
        tmp_path,
        [{"id": "danger-001", "subtype": "hardware_mapping", "question": "Do something risky."}],
    )
    seeds = load_seed_cases(p)

    bot = MagicMock()
    bot.ask.return_value = Answer(
        text="Submitted.",
        tool_calls=[ToolCall(name="ibm_submit_execution", input={}, output="{}")],
        confidence="medium",
    )
    judge_client = MagicMock()
    log_path = tmp_path / "logs" / "batch.jsonl"

    candidates = generate_candidates(
        seeds=seeds,
        bot=bot,
        judge_client=judge_client,
        judge_model="claude-haiku-4-5",
        log_path=log_path,
    )

    assert candidates == []
    judge_client.messages.create.assert_not_called()
    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["verdict"] == "skipped_unsafe_tool_invoked"


def test_generate_candidates_logs_judge_parse_error_and_continues(tmp_path: Path):
    p = _write_seeds(
        tmp_path,
        [{"id": "bad-json-001", "subtype": "definition", "question": "What is a stabilizer?"}],
    )
    seeds = load_seed_cases(p)

    bot = MagicMock()
    bot.ask.return_value = Answer(text="A stabilizer is...", confidence="medium")
    judge_client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text="not json")]
    judge_client.messages.create.return_value = msg
    log_path = tmp_path / "logs" / "batch.jsonl"

    candidates = generate_candidates(
        seeds=seeds,
        bot=bot,
        judge_client=judge_client,
        judge_model="claude-haiku-4-5",
        log_path=log_path,
    )

    assert candidates == []
    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["verdict"] == "judge_parse_error"
