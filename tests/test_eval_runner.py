"""Eval runner integration test with stubbed ChatQEC."""

from unittest.mock import MagicMock

from chatqec.api import Answer, Citation
from chatqec.eval.dataset import EvalCase, QuestionSubtype
from chatqec.eval.runner import EvalReport, run_eval


def _case(case_id: str, subtype: QuestionSubtype, expected_prefix: str) -> EvalCase:
    return EvalCase(
        id=case_id,
        subtype=subtype,
        question="q",
        expected_topics=[],
        expected_source_id_prefixes=[expected_prefix],
        gold_answer_summary="ref",
    )


def test_runner_aggregates_metrics_per_subtype():
    cases = [
        _case("d1", QuestionSubtype.DEFINITION, "canonical:surface-code"),
        _case("d2", QuestionSubtype.DEFINITION, "canonical:steane-code"),
        _case("p1", QuestionSubtype.PAPER_LOOKUP, "arxiv:2408.13687"),
    ]

    bot = MagicMock()

    def fake_ask(q):
        return Answer(
            text="Answer [^1]",
            citations=[
                Citation(
                    source_id="canonical:surface-code",
                    chunk_id="c1",
                    title="t",
                    url="u",
                )
            ],
            confidence="medium",
        )

    bot.ask.side_effect = fake_ask

    judge_client = MagicMock()
    judge_msg = MagicMock()
    judge_msg.content = [MagicMock(text="yes")]
    judge_client.messages.create.return_value = judge_msg

    report = run_eval(
        cases=cases,
        bot=bot,
        judge_client=judge_client,
        judge_model="claude-haiku-4-5",
    )

    assert isinstance(report, EvalReport)
    assert report.total == 3
    assert report.per_subtype["definition"].n == 2
    assert report.per_subtype["paper_lookup"].n == 1
    assert 0.0 < report.overall_recall_at_k <= 1.0


def test_runner_handles_empty_case_list():
    bot = MagicMock()
    judge = MagicMock()
    report = run_eval(cases=[], bot=bot, judge_client=judge, judge_model="claude-haiku-4-5")
    assert report.total == 0
    assert report.overall_recall_at_k == 0.0
