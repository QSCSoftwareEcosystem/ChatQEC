"""``chatqec eval bench``: run external QEC benchmarks through chatQEC and report them."""

from __future__ import annotations

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


def _bench(name: str) -> tuple[Benchmark, Path]:
    kwargs = _DATA[name]
    return get_benchmark(name)(**kwargs), next(iter(kwargs.values()))


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
        resolvable = sum(item.meta.get("resolvable", {}).values())
        click.echo(f"{item.id}\tresolvable={resolvable}/{len(item.gold)}")


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
) -> None:
    """Ask every item through the full chatQEC pipeline and score it."""
    from chatqec.cli import _build_chatqec
    from chatqec.conversation.memory import ConversationMemory
    from chatqec.eval.benchmarks import provenance
    from chatqec.eval.benchmarks.run import run_benchmark, summarize

    if resume and out_path is None:
        raise click.UsageError("--resume needs --out")
    bench, gold = _bench(name)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_path or REPO / f"eval/results/{name}_{model}_{stamp}.jsonl"
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
    bot = _build_chatqec(mcp_server=mcp_server)
    selected = set(ids.split(",")) if ids else None

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
