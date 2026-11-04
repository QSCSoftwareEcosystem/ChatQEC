"""Eval runner integration test with stubbed ChatQEC."""

from unittest.mock import MagicMock

from chatqec.api import Answer, Citation, ToolCall
from chatqec.eval.dataset import EvalCase, QuestionSubtype
from chatqec.eval.runner import EvalReport, run_eval


def _case(
    case_id: str,
    subtype: QuestionSubtype,
    expected_prefix: str,
    *,
    complexity: str = "simple",
    expected_tools: list[str] | None = None,
    target_capability: list[str] | None = None,
    is_adversarial: bool = False,
    expected_numeric_facts: dict | None = None,
) -> EvalCase:
    return EvalCase(
        id=case_id,
        subtype=subtype,
        question="q",
        expected_topics=[],
        expected_source_id_prefixes=[expected_prefix],
        gold_answer_summary="ref",
        complexity=complexity,
        expected_tools=expected_tools or [],
        target_capability=target_capability or [],
        is_adversarial=is_adversarial,
        expected_numeric_facts=expected_numeric_facts or {},
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
            retrieved_source_ids=["canonical:surface-code"],  # a cited source was retrieved
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


def _judge_client() -> MagicMock:
    judge_client = MagicMock()
    judge_msg = MagicMock()
    judge_msg.content = [MagicMock(text="yes")]
    judge_client.messages.create.return_value = judge_msg
    return judge_client


def test_runner_computes_tool_call_accuracy_when_expected_tools_present():
    case = _case(
        "route-1",
        QuestionSubtype.CODE_EXECUTION,
        "",
        expected_tools=["stim_simulate"],
        target_capability=["stim"],
    )
    bot = MagicMock()
    bot.ask.return_value = Answer(
        text="Simulated.",
        tool_calls=[ToolCall(name="stim_simulate", input={}, output="{}")],
        confidence="medium",
    )

    report = run_eval(
        cases=[case], bot=bot, judge_client=_judge_client(), judge_model="claude-haiku-4-5"
    )

    row = report.raw[0]
    assert row["tool_call_accuracy"]["f1"] == 1.0
    assert row["tool_calls"] == [{"name": "stim_simulate", "input": {}}]
    assert row["elapsed_seconds"] >= 0.0
    assert report.per_capability["stim"].tool_call_f1 == 1.0


def test_runner_leaves_tool_call_f1_none_when_not_applicable():
    case = _case("wiki-1", QuestionSubtype.DEFINITION, "", target_capability=["qappswiki"])
    bot = MagicMock()
    bot.ask.return_value = Answer(text="From the wiki.", confidence="medium")

    report = run_eval(
        cases=[case], bot=bot, judge_client=_judge_client(), judge_model="claude-haiku-4-5"
    )

    assert report.raw[0]["tool_call_accuracy"] == {}
    assert report.per_capability["qappswiki"].tool_call_f1 is None


def test_runner_buckets_by_complexity_and_adversarial():
    cases = [
        _case("s1", QuestionSubtype.DEFINITION, "", complexity="simple"),
        _case("c1", QuestionSubtype.CODE_EXECUTION, "", complexity="compound", is_adversarial=True),
    ]
    bot = MagicMock()
    bot.ask.return_value = Answer(text="A.", confidence="medium")

    report = run_eval(
        cases=cases, bot=bot, judge_client=_judge_client(), judge_model="claude-haiku-4-5"
    )

    assert report.per_complexity["simple"].n == 1
    assert report.per_complexity["compound"].n == 1
    assert report.adversarial_vs_non["adversarial"].n == 1
    assert report.adversarial_vs_non["non_adversarial"].n == 1


def test_runner_computes_numeric_fact_match():
    case = _case(
        "num-1",
        QuestionSubtype.CODE_EXECUTION,
        "",
        expected_numeric_facts={"n": 9, "k": 1},
    )
    bot = MagicMock()
    bot.ask.return_value = Answer(
        text="[[9,1,3]] code.",
        tool_calls=[ToolCall(name="code_params", input={}, output='{"n": 9, "k": 1}')],
        confidence="medium",
    )

    report = run_eval(
        cases=[case], bot=bot, judge_client=_judge_client(), judge_model="claude-haiku-4-5"
    )

    assert report.raw[0]["numeric_fact_match"] == {"n": True, "k": True}


def test_recall_uses_retrieved_sources_not_citations(tmp_path):
    """A source that was retrieved but not cited must count for recall@k."""
    from unittest.mock import MagicMock

    from chatqec.api_types import Answer
    from chatqec.eval.dataset import EvalCase
    from chatqec.eval.runner import run_eval

    case = EvalCase(
        id="c1", subtype="definition", question="q", expected_topics=[],
        expected_source_id_prefixes=["canonical:surface"], gold_answer_summary="g",
    )  # fmt: skip
    bot = MagicMock()
    bot.ask.return_value = Answer(
        text="a", citations=[], retrieved_source_ids=["canonical:surface-code"]
    )
    judge = MagicMock()
    report = run_eval(cases=[case], bot=bot, judge_client=judge, judge_model="m")
    assert report.overall_recall_at_k == 1.0
