"""ChatQEC-vs-bare comparison adapter: per-category scoring, citation checks, judge use."""

import json
from unittest.mock import MagicMock

import pytest

from chatqec.api_types import Answer
from chatqec.eval.benchmarks import get_benchmark
from chatqec.eval.benchmarks.compare import CompareBench, arxiv_ids, code_triples

ITEMS = [
    {"id": "lit-bb5", "category": "literature", "prompt": "List BB5 codes.",
     "gold": {"codes": [[30, 4, 5], [48, 4, 7]]}, "source": "arXiv:2503.22071"},
    {"id": "comp-db", "category": "computation", "prompt": "LERs?",
     "gold": {"rates": {"pymatching": 0.05, "belieffind": 0.119}, "shots": 20_000}, "source": "s"},
    {"id": "comp-t", "category": "computation", "prompt": "T count?",
     "gold": {"exact": {"t_count": 65}}, "source": "s"},
    {"id": "def-1", "category": "definition", "prompt": "What is the surface code?",
     "gold": {"summary": "A topological CSS code on a 2D lattice."}, "source": "s"},
    {"id": "abstain-1", "category": "abstain", "prompt": "Willow d=15 LER?",
     "gold": {"premise": "No d=15 result exists."}, "source": "s"},
]  # fmt: skip


def _judge(reply: str) -> MagicMock:
    judge = MagicMock()
    judge.messages.create.return_value = MagicMock(content=[MagicMock(text=reply)])
    return judge


@pytest.fixture
def bench_factory(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text("".join(json.dumps(i) + "\n" for i in ITEMS))

    def make(reply="yes", known=("2503.22071",), exists=lambda i: i == "2401.00001"):
        return CompareBench(
            prompts_jsonl=path, judge=_judge(reply), judge_model="gemma-4",
            known_arxiv_ids=set(known), arxiv_exists=exists,
        )  # fmt: skip

    return make


def _item(bench, item_id):
    return next(i for i in bench.items() if i.id == f"compare/{item_id}")


def test_registered():
    assert get_benchmark("compare") is CompareBench


def test_literature_needs_every_gold_code_and_no_invented_ones(bench_factory):
    b = bench_factory()
    s = b.score(_item(b, "lit-bb5"), Answer(text="[[30,4,5]] and [[48, 4, 7]] (arXiv:2503.22071)"))
    assert s["recall"] == 1.0 and s["invented"] == 0 and s["passed"]
    s = b.score(_item(b, "lit-bb5"), Answer(text="[[30,4,5]], [[72,12,6]]"))
    assert s["recall"] == 0.5 and s["invented"] == 1 and not s["passed"]


def test_computation_matches_stated_rates_against_gold(bench_factory):
    b = bench_factory()
    good = Answer(text="PyMatching: 0.0497, BeliefFind: 0.1200")
    bad = Answer(text="PyMatching: 0.01, BeliefFind: 0.30")
    assert b.score(_item(b, "comp-db"), good)["passed"]
    s = b.score(_item(b, "comp-db"), bad)
    assert s["rates_matched"] == 0.0 and not s["passed"]


def test_exact_values_must_appear(bench_factory):
    b = bench_factory()
    assert b.score(_item(b, "comp-t"), Answer(text="The T count is 65."))["passed"]
    assert not b.score(_item(b, "comp-t"), Answer(text="About 60 T gates."))["passed"]


def test_definition_and_abstain_use_the_judge(bench_factory):
    assert bench_factory("yes").score(
        _item(bench_factory(), "def-1"), Answer(text="A 2D topological code.")
    )["passed"]
    b = bench_factory("no")
    s = b.score(_item(b, "abstain-1"), Answer(text="It was 1e-9."))
    assert s["judge"] == "no" and not s["passed"]


def test_citations_are_classified_known_real_or_fabricated(bench_factory):
    b = bench_factory()
    s = b.score(
        _item(b, "def-1"),
        Answer(text="See arXiv:2503.22071, arXiv:2401.00001 and arXiv:2499.99999."),
    )
    assert s["cited"] == 3 and s["cited_in_corpus"] == 1 and s["fabricated"] == 1


def test_helpers_parse_codes_and_ids():
    assert code_triples("[[30,4,5]] and [[ 48 , 4 , 7 ]]") == {(30, 4, 5), (48, 4, 7)}
    assert code_triples("a \\code{323,14,15} code and \u27e6144,12,12\u27e7") == {
        (323, 14, 15),
        (144, 12, 12),
    }
    assert arxiv_ids("arXiv:2503.22071v2 and https://arxiv.org/abs/2604.21287") == [
        "2503.22071",
        "2604.21287",
    ]


def test_arxiv_checker_parses_feeds_and_caches(tmp_path):
    from chatqec.eval.benchmarks.arxiv_check import cached_checker, exists_from_feed

    real = (
        "<feed><entry><id>http://arxiv.org/abs/2503.22071v2</id><title>QEC</title></entry></feed>"
    )
    error = (
        "<feed><entry><id>http://arxiv.org/api/errors#bad</id><title>Error</title></entry></feed>"
    )
    assert exists_from_feed(real, "2503.22071") and not exists_from_feed(error, "2499.99999")
    calls = []

    def fetch(i):
        calls.append(i)
        return real if i == "2503.22071" else error

    cache = tmp_path / "arxiv.json"
    check = cached_checker(cache, fetch)
    assert check("2503.22071") and not check("2499.99999") and check("2503.22071")
    assert calls == ["2503.22071", "2499.99999"]
    assert cached_checker(cache, fetch)("2499.99999") is False and len(calls) == 2


def test_codes_the_paper_itself_mentions_are_not_invented(tmp_path):
    """The BB5 paper compares against BB6 codes ([[30,4,4]], [[48,4,6]]); quoting that
    comparison is correct, not an invention (the first smoke run penalised it)."""
    rows = [{"id": "lit", "category": "literature", "prompt": "p", "source": "arXiv:2503.22071",
             "gold": {"codes": [[30, 4, 5], [48, 4, 7]],
                      "mentioned": [[30, 4, 4], [30, 4, 5], [48, 4, 6], [48, 4, 7]]}}]  # fmt: skip
    path = tmp_path / "p.jsonl"
    path.write_text(json.dumps(rows[0]) + "\n")
    b = CompareBench(prompts_jsonl=path, judge=None, judge_model="", known_arxiv_ids=set(),
                     arxiv_exists=lambda i: False)  # fmt: skip
    item = b.items()[0]
    quoted = Answer(text="BB5: [[30,4,5]], [[48,4,7]]; BB6 only reach [[30,4,4]] and [[48,4,6]].")
    assert b.score(item, quoted)["passed"]
    made_up = Answer(text="[[30,4,5]], [[48,4,7]] and the flagship [[144,12,12]].")
    s = b.score(item, made_up)
    assert s["invented"] == 1 and not s["passed"]


def test_code_triples_reads_latex_double_bracket_forms():
    """OmniQEC (arXiv:2607.25865) writes $[\\![72,12,6]\\!]$; papers also use \\llbracket.
    Missing these left a paper's 'mentioned' set empty, so a correct answer quoting the
    paper's codes was scored as inventing them."""
    assert code_triples(r"BB $[\![72,12,6]\!]$ and $[\![144,12,12]\!]$") == {
        (72, 12, 6),
        (144, 12, 12),
    }
    assert code_triples(r"$\llbracket 90, 8, 10 \rrbracket$") == {(90, 8, 10)}
