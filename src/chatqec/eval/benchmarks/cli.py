"""``chatqec eval bench``: run external QEC benchmarks through chatQEC and report them."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import yaml

from chatqec.eval.benchmarks import get_benchmark
from chatqec.eval.benchmarks.base import Benchmark

REPO = Path(__file__).resolve().parents[4]
PINS = REPO / "eval/benchmarks/pins.yaml"
# Adapter name -> constructor arguments and the gold file recorded in provenance.
_DATA: dict[str, dict[str, Path]] = {
    "decoder-bench": {"gold_csv": REPO / "eval/benchmarks/decoder_bench/benchmark_results.csv"},
    "ft-primitive-bench": {"gold_jsonl": REPO / "eval/benchmarks/ft_primitive_bench/gold.jsonl"},
}
COMPARE_PROMPTS = REPO / "eval/compare/prompts.jsonl"
# Native counts for pairs decoder-bench's own pipeline does not reproduce (rescore --audited-gold).
AUDIT = REPO / "eval/benchmarks/decoder_bench/gold_audit.jsonl"
ARXIV_CACHE = REPO / "eval/compare/arxiv_exists.json"
# The bare baseline: the same model, no retrieval, no tools, a neutral expert prompt.
BARE_SYSTEM = (
    "You are a helpful assistant with expertise in quantum error correction. Answer accurately "
    "and concisely. Cite sources (for example arXiv ids) when you rely on them."
)


def _bench(name: str, judge_model: str | None = None) -> tuple[Benchmark, Path]:
    if name == "compare":
        from chatqec.cli import _make_judge_client
        from chatqec.eval.benchmarks.arxiv_check import cached_checker, corpus_arxiv_ids
        from chatqec.eval.benchmarks.compare import CompareBench

        judge = _make_judge_client(provider="local", model=judge_model) if judge_model else None
        bench = CompareBench(
            prompts_jsonl=COMPARE_PROMPTS,
            judge=judge,
            judge_model=judge_model or "",
            known_arxiv_ids=corpus_arxiv_ids(REPO / "sources/registry.jsonl"),
            arxiv_exists=cached_checker(ARXIV_CACHE),
        )
        return bench, COMPARE_PROMPTS
    kwargs = _DATA[name]
    return get_benchmark(name)(**kwargs), next(iter(kwargs.values()))


def _bare_ask(provider: str, model: str) -> Any:
    """Ask the model directly: no retrieval, no tools, no chatQEC system prompt."""
    from chatqec.api_types import Answer
    from chatqec.cli import _build_llm_client, _load_config

    local = _load_config().get("models", {}).get("local", {})
    client: Any = _build_llm_client(provider, local)

    def ask(prompt: str) -> Answer:
        reply = client.messages.create(
            model=model,
            max_tokens=4000,
            system=BARE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(getattr(b, "text", "") for b in reply.content)
        usage = getattr(reply, "usage", None)
        return Answer(
            text=text,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        )

    return ask


def _model_url(model: str) -> str | None:
    cfg = yaml.safe_load((REPO / "conf/config.yaml").read_text())
    local = (cfg.get("models") or {}).get("local") or {}
    for role in ("primary", "fallback"):
        if local.get(f"{role}_model") == model:
            return str(local.get(f"{role}_url"))
    return None


@click.group("bench")
def bench_group() -> None:
    """External QEC benchmarks (decoder-bench, FTPrimitiveBench) scored on their own gold."""


@bench_group.command("list")
@click.argument("name")
def bench_list(name: str) -> None:
    """List a benchmark's items."""
    bench, _ = _bench(name)
    for item in bench.items():
        flags = item.meta.get("resolvable", {})
        # per-decoder dict (decoder-bench) or one flag for the item's single rate
        values = list(flags.values()) if isinstance(flags, dict) else [bool(flags)]
        click.echo(f"{item.id}\tresolvable={sum(values)}/{len(values)}")


@bench_group.command("run")
@click.argument("name")
@click.option("--model", required=True, help="Served local model to pin, e.g. muse-glimmer")
@click.option("--provider", default="local", show_default=True)
@click.option("--mcp-server", default="chatqec-mcp-tools", show_default=True)
@click.option("--limit", type=int, default=None, help="Only the first N items (smoke runs)")
@click.option("--ids", default=None, help="Comma-separated item ids to run")
@click.option("--out", "out_path", type=click.Path(path_type=Path), default=None)
@click.option("--resume", is_flag=True, help="Append to --out, skipping items already in it")
@click.option("--tool-timeout", type=float, default=1800.0, show_default=True)
@click.option(
    "--system",
    type=click.Choice(["chatqec", "bare"]),
    default="chatqec",
    show_default=True,
    help="chatqec: full pipeline; bare: the same model with no retrieval and no tools",
)
@click.option(
    "--judge-model",
    default=None,
    help="compare only: local model that grades definition/abstain items (not the answering one)",
)
def bench_run(
    name: str,
    model: str,
    provider: str,
    mcp_server: str,
    limit: int | None,
    ids: str | None,
    out_path: Path | None,
    resume: bool,
    tool_timeout: float,
    system: str,
    judge_model: str | None,
) -> None:
    """Ask every item through chatQEC (or the bare model) and score it."""
    from chatqec.cli import _build_chatqec
    from chatqec.conversation.memory import ConversationMemory
    from chatqec.eval.benchmarks import provenance
    from chatqec.eval.benchmarks.run import run_benchmark, summarize

    if resume and out_path is None:
        raise click.UsageError("--resume needs --out")
    if name == "compare" and not judge_model:
        raise click.UsageError("compare needs --judge-model (a local model other than --model)")
    if judge_model and judge_model == model:
        raise click.UsageError("--judge-model must differ from --model: no model grades itself")
    bench, gold = _bench(name, judge_model)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"{system}-{model}" if name == "compare" else model
    out = out_path or REPO / f"eval/results/{name}_{label}_{stamp}.jsonl"
    os.environ["CHATQEC_PROVIDER"] = provider
    os.environ["CHATQEC_MCP_TIMEOUT_SECONDS"] = str(tool_timeout)
    if provider == "local":
        os.environ["CHATQEC_LOCAL_MODEL"] = model  # one model, no silent fallback
    settings: dict[str, Any] = {
        "benchmark": name,
        "provider": provider,
        "mcp_server": mcp_server,
        "tool_timeout_s": tool_timeout,
        "limit": limit,
        "ids": ids,
        "fresh_memory_per_item": True,
        "system": system,
        "judge_model": judge_model,
        "bare_system_prompt": BARE_SYSTEM if system == "bare" else None,
    }
    header = provenance.collect(
        chatqec_repo=REPO,
        tools_repo=Path(os.environ.get("CHATQEC_TOOLS_REPO", Path.home() / "chatqec-mcp-tools")),
        pins=PINS,
        gold=gold,
        model=model,
        model_url=_model_url(model) if provider == "local" else None,
        settings=settings,
    )
    selected = set(ids.split(",")) if ids else None
    if system == "bare":
        ask = _bare_ask(provider, model)
    else:
        bot = _build_chatqec(mcp_server=mcp_server)

        def ask(prompt: str) -> Any:
            return bot.ask(prompt, memory=ConversationMemory())  # items never share history

    def progress(row: dict[str, Any]) -> None:
        s = row["score"]
        click.echo(f"{row['id']}\tpassed={s.get('passed')}\t{row['seconds']}s\t{row['error']}")

    rows = run_benchmark(
        bench,
        ask,
        out,
        provenance=header,
        limit=limit,
        resume=resume,
        ids=selected,
        on_row=progress,
    )
    click.echo(f"{len(rows)} items -> {out}")
    click.echo(summarize(rows))


@bench_group.command("report")
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
def bench_report(paths: tuple[Path, ...]) -> None:
    """Summarize run files as a Markdown table (one row per file)."""
    from chatqec.eval.benchmarks.run import load_rows, summarize

    cols = (
        "n",
        "passed",
        "routed",
        "args_exact",
        "numeric_in_ci",
        "ranking_agreement",
        "faithful_best",
    )
    click.echo("| run | model | " + " | ".join(cols) + " |")
    click.echo("|" + "---|" * (len(cols) + 2))
    for path in paths:
        header, rows = load_rows(path)
        s = summarize(rows)
        cells = [f"{s[c]:.3f}" if isinstance(s[c], float) else str(s[c]) for c in cols]
        click.echo(f"| {path.name} | {header.get('model', '?')} | " + " | ".join(cells) + " |")


@bench_group.command("rescore")
@click.argument("name")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--out", "out_path", type=click.Path(path_type=Path), required=True)
@click.option(
    "--judge-model",
    default=None,
    help="compare only: re-judge with this local model (default: replay the stored verdicts)",
)
@click.option(
    "--audited-gold",
    is_flag=True,
    help="decoder-bench only: score the pairs decoder-bench's own pipeline does not reproduce "
    "against the audited native counts (eval/benchmarks/decoder_bench/gold_audit.jsonl)",
)
def bench_rescore(
    name: str, path: Path, out_path: Path, judge_model: str | None, audited_gold: bool
) -> None:
    """Rescore a stored run with the current scorer and gold (no answering-model calls)."""
    from chatqec.eval.benchmarks.decoder_bench import DecoderBench
    from chatqec.eval.benchmarks.provenance import sha256_file
    from chatqec.eval.benchmarks.run import load_rows, rescore_rows, summarize, write_rows

    if audited_gold and name != "decoder-bench":
        raise click.UsageError("--audited-gold applies to decoder-bench only")
    bench, gold = _bench(name, judge_model)
    if audited_gold:
        bench = DecoderBench(gold_csv=gold, audit_jsonl=AUDIT)
    replayed = name == "compare" and not judge_model
    if replayed:  # reuse the run's own judge verdicts instead of asking an LLM again
        from chatqec.eval.benchmarks.compare import CompareBench, ReplayJudge

        assert isinstance(bench, CompareBench)
        bench._judge = ReplayJudge.from_rows(load_rows(path)[1], bench)
        bench._judge_model = "replay"
    header, rows = rescore_rows(bench, path)
    if replayed:
        header["rescore_judge"] = "replay of the run's stored verdicts"
    header["rescore_gold"] = {"path": str(gold), "sha256": sha256_file(gold)}
    if audited_gold:
        header["rescore_audit"] = {"path": str(AUDIT), "sha256": sha256_file(AUDIT)}
    write_rows(out_path, header, rows)
    click.echo(f"{len(rows)} rows rescored -> {out_path}")
    click.echo(summarize(rows))


@bench_group.command("resume")
@click.argument("paths", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--retry-errors", is_flag=True, help="Move errored rows aside and ask those items again"
)
@click.option("--tool-timeout", type=float, default=None, help="Override the run's tool timeout")
@click.option("--allow-drift", is_flag=True, help="Resume although code or tool env changed")
@click.option("--dry-run", is_flag=True, help="Only report what each run still needs")
def bench_resume(
    paths: tuple[Path, ...],
    retry_errors: bool,
    tool_timeout: float | None,
    allow_drift: bool,
    dry_run: bool,
) -> None:
    """Continue unfinished runs with the settings in their own headers, one at a time.

    Without PATHS, every eval/results/*.jsonl run is checked. A run is skipped when it is
    complete, when another process is writing it, when its model is not being served, or
    (unless --allow-drift) when the answer-path code or the tool environment changed since
    it started. Each resume is noted in logs/bench_resume_notes.txt.
    """
    import subprocess
    import sys
    from datetime import UTC

    from chatqec.eval.benchmarks.provenance import served_model, tool_env_versions
    from chatqec.eval.benchmarks.resume import (
        ask_path_diff,
        drift_reasons,
        is_being_written,
        live_cmdlines,
        plan_run,
        resume_argv,
        split_errors,
    )

    runs = list(paths) or sorted(
        p
        for p in (REPO / "eval/results").glob("*.jsonl")
        if not p.name.endswith((".rescored.jsonl", ".errors.jsonl"))
    )
    env_now = tool_env_versions()
    notes = REPO / "logs/bench_resume_notes.txt"
    for path in runs:
        try:
            header_settings = json.loads(path.read_text().splitlines()[0])["_header"]["settings"]
            bench, _ = _bench(header_settings["benchmark"], header_settings.get("judge_model"))
            plan = plan_run(path, item_ids=[i.id for i in bench.items()])
        except (KeyError, ValueError, IndexError) as exc:
            click.echo(f"SKIP {path.name}: not a benchmark run ({exc})")
            continue
        status = f"{len(plan.done)} done, {len(plan.errored)} errored, {len(plan.missing)} missing"
        if plan.complete or (not plan.missing and not retry_errors):
            click.echo(f"OK   {path.name}: {status}")
            continue
        if is_being_written(path, live_cmdlines()):
            click.echo(f"BUSY {path.name}: {status}; another process is writing it")
            continue
        url = _model_url(plan.model) if plan.settings.get("provider", "local") == "local" else None
        if url is not None and not any(
            m.get("id") == plan.model for m in served_model(url) if isinstance(m, dict)
        ):
            click.echo(f"WAIT {path.name}: {status}; model {plan.model} is not served at {url}")
            continue
        reasons = drift_reasons(
            plan.header,
            tool_env=env_now,
            ask_path_diff=ask_path_diff(REPO, plan.header["chatqec"]["commit"]),
        )
        if reasons and not allow_drift:
            click.echo(f"DRIFT {path.name}: {status}; not resuming:\n  " + "\n  ".join(reasons))
            continue
        argv = resume_argv(plan, tool_timeout=tool_timeout)
        click.echo(f"RUN  {path.name}: {status}\n  {' '.join(argv)}")
        if dry_run:
            continue
        moved = split_errors(path) if retry_errors else 0
        notes.parent.mkdir(exist_ok=True)
        with notes.open("a") as f:
            f.write(
                f"=== RESUMED {datetime.now(UTC).isoformat(timespec='seconds')} {path.name} at "
                f"chatQEC {provenance_commit(REPO)}; {status}; errored rows moved aside: {moved}; "
                f"drift: {'; '.join(reasons) or 'none'}; argv: {' '.join(argv)}\n"
            )
        subprocess.run([sys.executable, "-m", "chatqec.cli", *argv[1:]], check=False)


def provenance_commit(repo: Path) -> str:
    from chatqec.eval.benchmarks.provenance import git_state

    state = git_state(repo)
    return f"{str(state.get('commit', '?'))[:7]}{' (dirty)' if state.get('dirty') else ''}"
