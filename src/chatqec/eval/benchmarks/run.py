"""Run a benchmark through chatQEC and record everything needed to reproduce and audit it.

Output is JSON Lines: a ``{"_header": provenance}`` line, then one row per item with the
prompt, the verbatim answer, every tool call (name, arguments, raw output), the score, the
wall time and token counts. Paper tables are derived from these rows only, and quoted
answers are copied from ``answer_text``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chatqec.api_types import Answer, ToolCall
from chatqec.eval.benchmarks.base import Benchmark

Ask = Callable[[str], Answer]
_SUMMED = ("passed", "routed", "args_exact", "tool_ok", "faithful_best")
_AVERAGED = ("numeric_in_ci", "ranking_agreement")


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if "id" in row:
            ids.add(row["id"])
    return ids


def run_benchmark(
    bench: Benchmark,
    ask: Ask,
    out: Path,
    *,
    provenance: dict[str, Any],
    limit: int | None = None,
    resume: bool = False,
    ids: set[str] | None = None,
    on_row: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Ask every (selected) item once and append its row to ``out``."""
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(out) if resume else set()
    items = [i for i in bench.items() if (ids is None or i.id in ids) and i.id not in done]
    if limit is not None:
        items = items[:limit]
    rows = []
    with out.open("a" if resume and out.exists() else "w") as f:
        if not (resume and done):
            f.write(json.dumps({"_header": provenance}) + "\n")
        for item in items:
            start = time.perf_counter()
            try:
                answer = ask(item.prompt)
                error = ""
                score = bench.score(item, answer)
            except Exception as exc:  # one bad item must not end a long run
                answer = Answer(text="")
                error = f"{type(exc).__name__}: {exc}"
                score = {"passed": False}
            row = {
                "id": item.id,
                "prompt": item.prompt,
                "expected_tool": item.expected_tool,
                "expected_args": item.expected_args,
                "gold": item.gold,
                "meta": item.meta,
                "answer_text": answer.text,
                "tool_calls": [
                    {"name": c.name, "input": c.input, "output": c.output}
                    for c in answer.tool_calls
                ],
                "score": score,
                "seconds": round(time.perf_counter() - start, 2),
                "input_tokens": answer.input_tokens,
                "output_tokens": answer.output_tokens,
                "error": error,
            }
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
            rows.append(row)
            if on_row is not None:
                on_row(row)
    return rows


def load_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header: dict[str, Any] = {}
    rows = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if "_header" in row:
            header = row["_header"]
        else:
            rows.append(row)
    return header, rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Rates over all rows for boolean scores; means over applicable rows for the rest."""
    n = len(rows)
    out: dict[str, Any] = {"n": n}
    for key in _SUMMED:
        out[key] = sum(bool(r["score"].get(key)) for r in rows) / n if n else 0.0
    for key in _AVERAGED:
        vals = [r["score"][key] for r in rows if r["score"].get(key) is not None]
        out[key] = sum(vals) / len(vals) if vals else None
        out[f"{key}_n"] = len(vals)
    return out


def rescore_rows(bench: Benchmark, path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score stored rows again with the current adapter and gold, without asking any model.

    Each row keeps its verbatim answer and tool calls, so a corrected gold value or scorer
    applies to an existing run. The header records when it was rescored.
    """
    header, rows = load_rows(path)
    items = {i.id: i for i in bench.items()}
    for row in rows:
        answer = Answer(
            text=row["answer_text"],
            tool_calls=[ToolCall(c["name"], c["input"], c["output"]) for c in row["tool_calls"]],
        )
        item = items.get(row["id"])
        if item is not None and not row.get("error"):
            row["score"] = bench.score(item, answer)
            row["gold"] = item.gold
            row["meta"] = item.meta
    header = {**header, "rescored_utc": datetime.now(UTC).isoformat(timespec="seconds")}
    return header, rows


def write_rows(path: Path, header: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [json.dumps({"_header": header})] + [json.dumps(r, default=str) for r in rows]
    path.write_text("\n".join(lines) + "\n")
