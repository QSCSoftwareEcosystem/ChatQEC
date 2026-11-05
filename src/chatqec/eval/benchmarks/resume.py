"""Resume benchmark runs that stopped before every item had a clean row.

A run file is JSON Lines: a ``_header`` (provenance and the run's settings) and one row per
item, appended and flushed as each item finishes, so a crash never loses finished items.
``chatqec eval bench run --resume`` skips every id already in the file, including rows that
ended in an error (for example a tool timeout). This module works out what a run still lacks,
moves errored rows aside when they should be retried, and refuses to resume when the code on
the answer path or the tool environment changed since the run started, because the resumed
rows would then not be comparable with the rows already in the file.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Code whose change alters how a question is answered. A diff here since the run's commit
# means resumed rows could differ from the existing ones for reasons other than the model.
ASK_PATH = (
    "src/chatqec/api.py",
    "src/chatqec/conversation",
    "src/chatqec/retrieval",
    "src/chatqec/providers",
    "src/chatqec/embedding",
    "src/chatqec/store",
    "conf",
)


@dataclass(frozen=True)
class RunPlan:
    path: Path
    header: dict[str, Any]
    done: list[str]
    errored: list[str]
    missing: list[str]

    @property
    def settings(self) -> dict[str, Any]:
        return dict(self.header.get("settings") or {})

    @property
    def benchmark(self) -> str:
        return str(self.settings["benchmark"])

    @property
    def model(self) -> str:
        return str(self.header["model"])

    @property
    def complete(self) -> bool:
        return not self.errored and not self.missing


def read_run(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if not lines or "_header" not in lines[0]:
        raise ValueError(f"{path} has no _header line")
    return lines[0]["_header"], lines[1:]


def plan_run(path: Path, *, item_ids: Sequence[str]) -> RunPlan:
    """What ``path`` still needs, given the benchmark's item ids in order."""
    header, rows = read_run(path)
    selected = (header.get("settings") or {}).get("ids")
    if selected:
        wanted = set(str(selected).split(","))
        item_ids = [i for i in item_ids if i in wanted]
    clean = {r["id"] for r in rows if "id" in r and not r.get("error")}
    errored = {r["id"] for r in rows if "id" in r and r.get("error")} - clean
    return RunPlan(
        path=path,
        header=header,
        done=[i for i in item_ids if i in clean],
        errored=[i for i in item_ids if i in errored],
        missing=[i for i in item_ids if i not in clean and i not in errored],
    )


def split_errors(path: Path) -> int:
    """Move errored rows to ``<stem>.errors.jsonl`` (appended, never deleted); return the count."""
    header, rows = read_run(path)
    bad = [r for r in rows if r.get("error")]
    if not bad:
        return 0
    aside = path.with_name(f"{path.stem}.errors.jsonl")
    with aside.open("a") as f:
        for r in bad:
            f.write(json.dumps(r, default=str) + "\n")
    keep = [json.dumps({"_header": header})] + [
        json.dumps(r, default=str) for r in rows if not r.get("error")
    ]
    path.write_text("\n".join(keep) + "\n")
    return len(bad)


def drift_reasons(
    header: dict[str, Any], *, tool_env: dict[str, str], ask_path_diff: str
) -> list[str]:
    """Why resuming would mix incomparable rows; empty when it is safe."""
    reasons = []
    for pkg, then in (header.get("tool_env") or {}).items():
        now = tool_env.get(pkg)
        if pkg != "python" and now != then:
            reasons.append(f"tool env changed: {pkg} {then} -> {now}")
    if ask_path_diff.strip():
        reasons.append(
            "answer-path code changed since the run's commit:\n" + ask_path_diff.rstrip()
        )
    return reasons


def ask_path_diff(repo: Path, since_commit: str) -> str:
    """``git diff --stat`` of the answer-path code between the run's commit and the work tree."""
    out = subprocess.run(
        ["git", "-C", str(repo), "diff", "--stat", since_commit, "--", *ASK_PATH],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


def is_being_written(path: Path, cmdlines: Iterable[Sequence[str]]) -> bool:
    """True when a live ``eval bench run`` process has ``path`` as its ``--out``."""
    target = str(path.resolve())
    for argv in cmdlines:
        if "bench" not in argv or "run" not in argv:
            continue
        for i, arg in enumerate(argv[:-1]):
            if arg == "--out" and str(Path(argv[i + 1]).resolve()) == target:
                return True
    return False


def live_cmdlines() -> list[list[str]]:
    """Command lines of this user's processes (Linux /proc)."""
    out = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            raw = (proc / "cmdline").read_bytes()
        except OSError:
            continue
        if raw:
            out.append([a.decode(errors="replace") for a in raw.split(b"\0") if a])
    return out


def resume_argv(plan: RunPlan, *, tool_timeout: float | None) -> list[str]:
    """The ``chatqec eval bench run`` call that continues ``plan`` with its own settings."""
    s = plan.settings
    argv = [
        "chatqec", "eval", "bench", "run", plan.benchmark,
        "--model", plan.model,
        "--provider", str(s.get("provider", "local")),
        "--mcp-server", str(s.get("mcp_server", "chatqec-mcp-tools")),
        "--system", str(s.get("system") or "chatqec"),
        "--tool-timeout", str(tool_timeout if tool_timeout is not None else s.get("tool_timeout_s", 1800.0)),
        "--out", str(plan.path),
        "--resume",
    ]  # fmt: skip
    if s.get("ids"):
        argv += ["--ids", str(s["ids"])]
    if s.get("judge_model"):
        argv += ["--judge-model", str(s["judge_model"])]
    return argv
