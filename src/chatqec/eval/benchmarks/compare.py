"""ChatQEC-vs-bare-LLM comparison: the same prompts, scored per category.

Items come from ``eval/compare/prompts.jsonl`` (built by scripts/benchmarks/build_compare_set.py
from gold already in the repo). The system under test is whatever ``ask`` the runner is given:
full chatQEC, or the same model with no retrieval and no tools. Scoring reads the answer text,
because a bare model has no tool calls:

  literature   every gold [[n,k,d]] stated, and none the paper never mentions (invented)
  computation  each gold rate matched by a stated number consistent with it at the item's shots
               (the gold lies in that number's 95% Wilson interval); exact values must appear
  definition   a local judge: consistent with the reviewed gold summary, no contradiction
  abstain      a local judge: does not invent the value and says it is not reported

Every arXiv id an answer cites is classified as in the corpus, real (exists on arXiv), or
fabricated. The judge is a local model named by the caller, so scoring spends no API credits.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from chatqec.api_types import Answer
from chatqec.eval.benchmarks import register_benchmark
from chatqec.eval.benchmarks.base import BenchItem, Benchmark
from chatqec.eval.benchmarks.decoder_bench import wilson
from chatqec.eval.benchmarks.ft_primitive_bench import _numbers

# [[n,k,d]], the LaTeX macro \code{n,k,d} some papers use, and the white square brackets
# U+27E6/U+27E7 that models also write.
_TRIPLE = re.compile(
    # [[n,k,d]], LaTeX [\![n,k,d]\!] and \llbracket n,k,d \rrbracket, \code{n,k,d}, and \u27e6n,k,d\u27e7
    r"(?:\[\[|\[\\!\[|\\llbracket|\\code\{|\u27e6)\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*"
    r"(?:\]\]|\]\\!\]|\\rrbracket|\}|\u27e7)"
)
_ARXIV = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(?:v\d+)?(?![\d])")
_JUDGE_SYSTEM = "You grade answers. Reply with exactly one word: yes or no."
_JUDGE = {
    "definition": (
        "Reference answer: {gold}\n\nCandidate answer: {answer}\n\nIs the candidate consistent "
        "with the reference, with no statement that contradicts it?"
    ),
    "abstain": (
        "The question has a false premise: {gold}\n\nQuestion: {prompt}\n\nCandidate answer: "
        "{answer}\n\nDoes the candidate avoid inventing the requested value, and say that it is "
        "not reported or does not exist?"
    ),
}


def code_triples(text: str) -> set[tuple[int, int, int]]:
    return {(int(a), int(b), int(c)) for a, b, c in _TRIPLE.findall(text)}


def arxiv_ids(text: str) -> list[str]:
    return list(dict.fromkeys(_ARXIV.findall(text)))


@register_benchmark("compare")
class CompareBench(Benchmark):
    def __init__(
        self,
        prompts_jsonl: Path | str,
        *,
        judge: Any,
        judge_model: str,
        known_arxiv_ids: set[str],
        arxiv_exists: Callable[[str], bool],
    ) -> None:
        lines = Path(prompts_jsonl).read_text().splitlines()
        self._rows = [json.loads(line) for line in lines if line]
        self._judge, self._judge_model = judge, judge_model
        self._known, self._exists = known_arxiv_ids, arxiv_exists

    def items(self) -> list[BenchItem]:
        return [
            BenchItem(
                id=f"compare/{r['id']}",
                prompt=r["prompt"],
                expected_tool="",
                expected_args={},
                gold=r["gold"],
                meta={"category": r["category"], "source": r["source"]},
            )
            for r in self._rows
        ]

    def score(self, item: BenchItem, answer: Answer) -> dict[str, Any]:
        category = item.meta["category"]
        scorer = {
            "literature": self._literature,
            "computation": self._computation,
        }.get(category, self._judged)
        out = scorer(item, answer)
        return {"category": category, **out, **self._citations(answer.text)}

    def _literature(self, item: BenchItem, answer: Answer) -> dict[str, Any]:
        gold = {tuple(c) for c in item.gold["codes"]}
        # Codes the paper itself mentions (e.g. the baselines it compares against) are not
        # inventions when an answer quotes them; anything else attributed to it is.
        mentioned = gold | {tuple(c) for c in item.gold.get("mentioned", [])}
        stated = code_triples(answer.text)
        recall = len(stated & gold) / len(gold)
        invented = len(stated - mentioned)
        return {"recall": recall, "invented": invented, "passed": recall == 1.0 and invented == 0}

    def _computation(self, item: BenchItem, answer: Answer) -> dict[str, Any]:
        numbers = _numbers(answer.text)
        if "exact" in item.gold:
            hits = [any(n == v for n in numbers) for v in item.gold["exact"].values()]
            return {"exact_matched": sum(hits) / len(hits), "passed": all(hits)}
        shots = int(item.gold["shots"])

        def consistent(gold: float) -> bool:
            for n in numbers:
                if 0 <= n <= 1:
                    lo, hi = wilson(round(n * shots), shots)
                    if lo <= gold <= hi:
                        return True
            return False

        hits = [consistent(g) for g in item.gold["rates"].values()]
        return {"rates_matched": sum(hits) / len(hits), "passed": all(hits)}

    def _judged(self, item: BenchItem, answer: Answer) -> dict[str, Any]:
        category = item.meta["category"]
        gold = item.gold.get("summary") or item.gold.get("premise")
        prompt = _JUDGE[category].format(gold=gold, prompt=item.prompt, answer=answer.text)
        reply = self._judge.messages.create(
            model=self._judge_model,
            max_tokens=5,
            system=_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(getattr(b, "text", "") for b in reply.content).strip().lower()
        verdict = "yes" if text.startswith("yes") else "no"
        return {"judge": verdict, "passed": verdict == "yes"}

    def _citations(self, text: str) -> dict[str, int]:
        ids = arxiv_ids(text)
        in_corpus = [i for i in ids if i in self._known]
        fabricated = [i for i in ids if i not in self._known and not self._exists(i)]
        return {"cited": len(ids), "cited_in_corpus": len(in_corpus), "fabricated": len(fabricated)}
