"""Build the ChatQEC-vs-bare-LLM comparison set from verifiable sources only.

Run in the chatQEC environment; the literature table is read from the installed tool server
(its own interpreter, in a subprocess):

    uv run python scripts/benchmarks/build_compare_set.py .

Writes eval/compare/prompts.jsonl. Every gold value comes from a file already in the repo:
  literature   chatqec_mcp_tools/tools/_paper_codes.py (each row checked against the paper text)
  computation  decoder-bench shipped gold and FTPrimitiveBench gold (eval/benchmarks/), and the
               slide's rz(0.3) T count, which ibm_ft_compile reproduces deterministically
  definition   eval/qa_set.jsonl gold summaries (hand reviewed)
  abstain      false-premise questions: the right answer says the value is not reported
Nothing is typed in by hand except the abstain questions and their premise.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from chatqec.eval.benchmarks.compare import code_triples
from chatqec.eval.benchmarks.ft_primitive_bench import FtPrimitiveBench

_DUMP_CODES = (
    "import json, chatqec_mcp_tools.tools._paper_codes as pc\n"
    "rows = [r for g in vars(pc).values() if isinstance(g, tuple) for r in g if hasattr(r, 'family')]\n"
    "print(json.dumps([[r.family, r.n, r.k, r.d, r.source] for r in rows]))\n"
)


def _paper_codes() -> list[list]:
    exe = shutil.which("chatqec-mcp-tools")
    if exe is None:
        raise SystemExit("chatqec-mcp-tools is not installed (uv tool install ...)")
    python = Path(exe).read_text().splitlines()[0][2:].strip()
    out = subprocess.run(
        [python, "-I", "-c", _DUMP_CODES], capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)


DB_ITEMS = [  # (code, d, p, decoders): fast decoders, all resolvable at 20k shots
    ("surface", 7, 0.005, ("pymatching", "belieffind")),
    ("surface", 11, 0.005, ("pymatching", "belieffind")),
    ("color", 7, 0.003, ("belieffind", "bplsd")),
    ("qldpc", 6, 0.003, ("belieffind", "bplsd")),
]
FT_ITEMS = [
    "mem-d5-uniform-p3e-3",
    "mem-d5-zbias10-p3e-3",
    "th-d5-zbias10-p3e-3",
    "ls-d5-uniform-p3e-3",
]
ABSTAIN = [
    (
        "abstain-willow-d15",
        "What logical error per cycle did Google report for a distance-15 surface code memory "
        "on its Willow processor?",
        "Google's Willow memory experiments go up to distance 7; no distance-15 result exists.",
    ),
    (
        "abstain-fake-arxiv",
        "What are the [[n,k,d]] parameters of the code introduced in arXiv:2607.99999?",
        "arXiv:2607.99999 is not a real QEC paper in the corpus; the parameters are unknown.",
    ),
    (
        "abstain-bb5-d11",
        "What is the logical error rate of the [[144,12,11]] BB5 code at p = 0.001?",
        "No [[144,12,11]] BB5 code is reported in the BB5 papers; no such value exists.",
    ),
]
_HUMAN = {"pymatching": "PyMatching", "belieffind": "BeliefFind", "bplsd": "BP+LSD"}
_CODE = {
    "surface": "rotated surface code",
    "color": "color code",
    "qldpc": "bivariate bicycle (BB) qLDPC code",
}
_FAMILY = {
    "bb5": "BB5 (weight-5 bivariate bicycle)",
    "tile": "tile",
    "directional_tile": "directional tile",
    "barbell": "barbell",
}


def _mentioned(repo: Path, source: str) -> list[list[int]]:
    """Every [[n,k,d]] in the paper's ingested full text (its baselines included)."""
    chunks = repo / "data/chunks" / f"arxiv_{source.split(':')[1]}.jsonl"
    text = " ".join(json.loads(line).get("text", "") for line in chunks.read_text().splitlines())
    return [list(t) for t in sorted(code_triples(text))]


def _literature(repo: Path) -> list[dict]:
    groups: dict[tuple[str, str], list] = {}
    for family, n, k, d, source in _paper_codes():
        groups.setdefault((family, source), []).append((n, k, d))
    out = []
    for (family, source), triples in sorted(groups.items()):
        out.append({
            "id": f"lit-{family}-{source.split(':')[1]}",
            "category": "literature",
            "prompt": f"List the [[n,k,d]] parameters of the {_FAMILY[family]} codes reported in {source}.",
            "gold": {"codes": sorted(set(triples)), "mentioned": _mentioned(repo, source)},
            "source": source,
        })  # fmt: skip
    return out


def _db(repo: Path) -> list[dict]:
    gold: dict[tuple, dict] = {}
    rx = re.compile(r"(surface|color|qldpc)_circuit_(\d+)_([0-9.]+?)(?:_z)?\.h5$")
    with (repo / "eval/benchmarks/decoder_bench/benchmark_results.csv").open() as f:
        for row in csv.DictReader(f):
            m = rx.match(row["filename"])
            if m:
                gold.setdefault((m.group(1), int(m.group(2)), float(m.group(3))), {})[
                    row["decoder"]
                ] = float(row["logical_error_rate"])
    out = []
    for code, d, p, decs in DB_ITEMS:
        names = " and ".join(_HUMAN[x] for x in decs)
        out.append({
            "id": f"comp-db-{code}-d{d}-p{p}",
            "category": "computation",
            "prompt": (f"Using decoder-bench's SI1000 circuit noise, what logical error rates do {names} "
                       f"reach on a distance-{d} {_CODE[code]} memory experiment at p = {p}? "
                       "Use 20,000 shots and seed 42."),
            "gold": {"rates": {x: gold[(code, d, p)][x] for x in decs}, "shots": 20_000},
            "source": "decoder-bench shipped results (commit 9733a55)",
        })  # fmt: skip
    return out


def _ft(repo: Path) -> list[dict]:
    bench = FtPrimitiveBench(gold_jsonl=repo / "eval/benchmarks/ft_primitive_bench/gold.jsonl")
    by_id = {i.id.split("/")[1]: i for i in bench.items()}
    return [
        {
            "id": f"comp-ft-{k}",
            "category": "computation",
            "prompt": by_id[k].prompt,
            "gold": {
                "rates": {"logical_error_rate": by_id[k].gold["logical_error_rate"]},
                "shots": 20_000,
            },
            "source": "FTPrimitiveBench library at 200,000 shots (commit b75fbd1)",
        }
        for k in FT_ITEMS
    ]


def _fixed() -> list[dict]:
    return [{
        "id": "comp-rz03-tcount",
        "category": "computation",
        "prompt": "Compile rz(0.3) to Clifford+T at 6 digits of precision with gridsynth and report the T count.",
        "gold": {"exact": {"t_count": 65}},
        "source": "ibm_ft_compile (Qiskit 2.6 gridsynth), deterministic",
    }]  # fmt: skip


def _definitions(repo: Path) -> list[dict]:
    out = []
    for line in (repo / "eval/qa_set.jsonl").read_text().splitlines():
        case = json.loads(line)
        if case["subtype"] == "definition":
            out.append({
                "id": f"def-{case['id']}",
                "category": "definition",
                "prompt": case["question"],
                "gold": {"summary": case["gold_answer_summary"]},
                "source": "eval/qa_set.jsonl (reviewed gold summary)",
            })  # fmt: skip
    return out


def main(repo: Path) -> None:
    items = _literature(repo) + _db(repo) + _ft(repo) + _fixed() + _definitions(repo)
    items += [
        {
            "id": i,
            "category": "abstain",
            "prompt": q,
            "gold": {"premise": why},
            "source": "false premise",
        }
        for i, q, why in ABSTAIN
    ]
    out = repo / "eval/compare/prompts.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(i, default=list) + "\n" for i in items))
    print(len(items), "items ->", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
