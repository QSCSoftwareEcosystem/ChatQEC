"""Write the paper's result tables as LaTeX, straight from the run files (no typed numbers).

    uv run python scripts/make_paper_tables.py 6ac52ba22379cb285b6dc6d7/tables

Writes:
  bench.tex    decoder-bench (shipped and audited gold) and FTPrimitiveBench, per model
  compare.tex  ChatQEC versus the same model without retrieval or tools, by category
  sources.txt  which run file each column came from (for docs/benchmarks/REPRODUCE.md)

The paper \\input{}s these. Re-run after any rescore; every number traces to a file below.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from chatqec.eval.benchmarks.run import summarize

logger = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parent.parent
R = REPO / "eval/results"
MODELS = {"muse-glimmer": "Muse", "gemma-4": "Gemma"}

DB = {m: R / f"decoder-bench_{m}_20261104_173715.jsonl" for m in MODELS}
DB_AUDITED = {m: R / f"decoder-bench_{m}_20261104_173715.audited.jsonl" for m in MODELS}
FTPB = {
    "muse-glimmer": R / "ft-primitive-bench_muse-glimmer_20261105_153626.jsonl",
    "gemma-4": R / "ft-primitive-bench_gemma-4_20261105_222146.jsonl",
}
CMP = {
    (system, m): R / f"compare_{system}_{m}_20261105_222146.rescored.jsonl"
    for system in ("chatqec", "bare")
    for m in MODELS
}
CATEGORIES = [
    ("computation", "Computation"),
    ("literature", "Literature"),
    ("abstain", "False premise"),
    ("definition", "Definition"),
]


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines()[1:] if x.strip()]


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    """The same aggregate `chatqec eval bench report` prints: boolean scores are rates over all
    rows (a row that errored, e.g. a tool timeout, counts as not passed / not routed); the
    numeric and ranking scores are means over the rows where they apply."""
    return summarize(rows)[key]


def _pct(x: float | None) -> str:
    return "--" if x is None else f"{100 * x:.0f}"


def bench_table() -> str:
    db = {m: _rows(DB[m]) for m in MODELS}
    dba = {m: _rows(DB_AUDITED[m]) for m in MODELS}
    ft = {m: _rows(FTPB[m]) for m in MODELS}
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{decoder-bench} & \multicolumn{2}{c}{FTPrimitiveBench} \\",
        r" & Muse & Gemma & Muse & Gemma \\",
        r"\midrule",
    ]
    specs = [
        ("Routed", "routed", db, ft),
        ("Args exact", "args_exact", db, ft),
        ("Numeric (shipped)", "numeric_in_ci", db, ft),
        ("Numeric (audited)", "numeric_in_ci", dba, None),
        ("Ranking", "ranking_agreement", db, None),
        ("Passed", "passed", dba, ft),
    ]
    for label, key, left, right in specs:
        cells = [_pct(_mean(left[m], key)) for m in MODELS]
        cells += [_pct(_mean(right[m], key)) if right else "--" for m in MODELS]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    n_db, n_ft = len(db["muse-glimmer"]), len(ft["muse-glimmer"])
    errs = [sum(1 for r in rows if r.get("error")) for rows in (*db.values(), *ft.values())]
    lines += [r"\midrule", f"Items & {n_db} & {n_db} & {n_ft} & {n_ft}" + r" \\"]
    lines += ["Errored items & " + " & ".join(str(e) for e in errs) + r" \\"]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def compare_table() -> str:
    cells: dict[tuple[str, str], dict[str, tuple[int, int]]] = {}
    extra: dict[tuple[str, str], dict[str, int]] = {}
    for key, path in CMP.items():
        by: dict[str, list[bool]] = defaultdict(list)
        rows = _rows(path)
        for r in rows:
            by[r["score"]["category"]].append(bool(r["score"].get("passed")))
        cells[key] = {c: (sum(v), len(v)) for c, v in by.items()}
        extra[key] = {
            "invented": sum(int(r["score"].get("invented") or 0) for r in rows),
            "fabricated": sum(int(r["score"].get("fabricated") or 0) for r in rows),
            "total": sum(bool(r["score"].get("passed")) for r in rows),
            "n": len(rows),
        }
    cols = [("chatqec", m) for m in MODELS] + [("bare", m) for m in MODELS]
    lines = [  # two-level header so the table fits one column
        r"\begin{tabular}{l" + "c" * len(cols) + "}",
        r"\toprule",
        r" & \multicolumn{2}{c}{ChatQEC} & \multicolumn{2}{c}{Bare model} \\",
        " & " + " & ".join(MODELS[m] for _, m in cols) + r" \\",
        r"\midrule",
    ]
    for cat, label in CATEGORIES:
        row = [f"{cells[k][cat][0]}/{cells[k][cat][1]}" for k in cols]
        lines.append(f"{label} & " + " & ".join(row) + r" \\")
    lines.append(r"\midrule")
    lines.append(
        "Total & " + " & ".join(f"{extra[k]['total']}/{extra[k]['n']}" for k in cols) + r" \\"
    )
    lines.append("Invented codes & " + " & ".join(str(extra[k]["invented"]) for k in cols) + r" \\")
    lines.append(
        "Fabricated ids & " + " & ".join(str(extra[k]["fabricated"]) for k in cols) + r" \\"
    )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "6ac52ba22379cb285b6dc6d7/tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench.tex").write_text(bench_table())
    (out / "compare.tex").write_text(compare_table())
    used = [*DB.values(), *DB_AUDITED.values(), *FTPB.values(), *CMP.values()]
    (out / "sources.txt").write_text("".join(f"{p.relative_to(REPO)}\n" for p in used))
    logger.info("wrote %s/{bench,compare}.tex from %d run files", out, len(used))


if __name__ == "__main__":
    main()
