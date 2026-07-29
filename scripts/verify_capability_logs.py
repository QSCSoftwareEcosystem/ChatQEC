#!/usr/bin/env python3
"""Mechanically verify capability-check raw logs against their eval prompt files.

This exists because the same defect class was found three times by hand across
this branch's fix rounds: a capability-check doc claimed a prompt's *literal*
eval-file wording was tested, when the actual logged ``ask.question`` was
something else (a paraphrase, a QEC-framed rephrasing, or an undisclosed
prefix). Manual review kept missing this. This script instead:

1. Loads an eval prompt file (``eval/*.json``) and indexes prompts by ``id``.
2. Walks a raw-logs directory and extracts every ``ask.question`` value logged
   in each ``.txt`` trace file (a file may contain more than one trace).
3. Maps each log file back to a prompt id by stripping a known "non-literal"
   suffix (``-retry``, ``-rephrased``, ``-paraphrase``, ``-qec-framed``) from
   the filename stem. Files without such a suffix (including ``-literal`` and
   ``-reverify``, which explicitly claim to test literal wording) are checked
   for **exact** (``==``) string equality against the JSON field.
4. Reports MATCH / MISMATCH / INFO (non-literal by filename, not checked) for
   every question found, and exits non-zero if any file claiming literal
   wording fails the exact-equality check.

Usage:
    python3 scripts/verify_capability_logs.py \\
        --eval eval/tsim_prompts.json \\
        --logs docs/tsim_glcb_capability_check_raw_logs

    # Or check all the known pairs shipped in this repo:
    python3 scripts/verify_capability_logs.py --all
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Filename suffixes that mean "this log intentionally tests a rephrased /
# non-literal wording" -- these are informational only, not checked for exact
# equality against the eval file's literal question.
NON_LITERAL_SUFFIXES = ("-retry", "-rephrased", "-paraphrase", "-qec-framed")

# Known (eval file, raw-logs dir) pairs in this repo, for `--all`.
KNOWN_PAIRS = [
    ("eval/tsim_prompts.json", "docs/tsim_glcb_capability_check_raw_logs"),
    ("eval/lightstim_qec_prompts.json", "docs/lightstim_capability_check_raw_logs"),
]


def load_questions_by_id(eval_path: Path) -> dict[str, str]:
    data = json.loads(eval_path.read_text())
    return {p["id"]: p["question"] for p in data["prompts"] if "question" in p}


def extract_ask_questions(log_text: str) -> list[str]:
    """Return every logged `ask.question = ...` value in a trace file.

    Handles both the single-line form (`question = <value>` immediately
    followed by `top_k = `) and the multi-line indented-block form the CLI
    emits when the question itself contains embedded newlines.
    """
    questions = []
    for m in re.finditer(r"ask\.question\n\s*question = (.*?)\n\s*top_k = ", log_text, re.DOTALL):
        raw = m.group(1)
        lines = raw.split("\n")
        if len(lines) == 1:
            questions.append(lines[0])
        else:
            stripped = [line.strip() for line in lines if line.strip() != ""]
            questions.append("\n".join(stripped))
    return questions


def prompt_id_for_filename(stem: str, known_ids: set[str]) -> tuple[str, bool]:
    """Map a log filename stem to a prompt id.

    Returns (prompt_id, is_literal_claim). `is_literal_claim` is False when the
    stem carries a known non-literal suffix (so mismatches there are expected
    and not reported as errors).
    """
    if stem in known_ids:
        return stem, True
    for suffix in NON_LITERAL_SUFFIXES:
        if stem.endswith(suffix):
            base = stem[: -len(suffix)]
            if base in known_ids:
                return base, False
    # Suffixes that still claim literal wording (e.g. "-literal", "-reverify")
    for suffix in ("-literal", "-reverify"):
        if stem.endswith(suffix):
            base = stem[: -len(suffix)]
            if base in known_ids:
                return base, True
    return stem, True


def verify_pair(eval_path: Path, logs_dir: Path) -> int:
    questions = load_questions_by_id(eval_path)
    known_ids = set(questions)
    failures = 0
    checked = 0

    for log_file in sorted(logs_dir.glob("*.txt")):
        stem = log_file.stem
        prompt_id, is_literal_claim = prompt_id_for_filename(stem, known_ids)
        if prompt_id not in questions:
            print(f"  ? {log_file.name}: no matching prompt id '{prompt_id}' in {eval_path.name}")
            continue

        expected = questions[prompt_id]
        found = extract_ask_questions(log_file.read_text())
        if not found:
            print(f"  ? {log_file.name}: no ask.question found in log")
            continue

        for q in found:
            checked += 1
            if q == expected:
                print(f"  OK  {log_file.name} [{prompt_id}]: matches eval file exactly")
            elif is_literal_claim:
                failures += 1
                print(f"  FAIL {log_file.name} [{prompt_id}]: claims literal wording but differs")
                print(f"       eval file: {expected!r}")
                print(f"       logged   : {q!r}")
            else:
                print(f"  INFO {log_file.name} [{prompt_id}]: non-literal by filename (expected)")

    print(f"{logs_dir}: {checked} question(s) checked, {failures} failure(s)")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, help="Path to eval/*.json prompt file")
    parser.add_argument("--logs", type=Path, help="Path to the matching raw-logs directory")
    parser.add_argument(
        "--all", action="store_true", help="Check all known (eval, logs) pairs in this repo"
    )
    args = parser.parse_args()

    total_failures = 0
    if args.all:
        for eval_str, logs_str in KNOWN_PAIRS:
            eval_path, logs_dir = Path(eval_str), Path(logs_str)
            if not logs_dir.exists():
                print(f"(skip: {logs_dir} does not exist)")
                continue
            print(f"=== {eval_path} vs {logs_dir} ===")
            total_failures += verify_pair(eval_path, logs_dir)
    elif args.eval and args.logs:
        total_failures += verify_pair(args.eval, args.logs)
    else:
        parser.error("pass --all, or both --eval and --logs")

    return 1 if total_failures else 0


if __name__ == "__main__":
    sys.exit(main())
