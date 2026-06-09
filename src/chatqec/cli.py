"""Click-based CLI entry point."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from chatqec import __version__
from chatqec.api import ChatQEC
from chatqec.canonical.loader import (
    CanonicalLoadError,
    load_canonical_directory,
)
from chatqec.ingestion.chunk import ChunkSink
from chatqec.ingestion.fetchers.arxiv import (
    ArxivMetadata,
)
from chatqec.ingestion.fetchers.arxiv import (
    paper_to_chunks as arxiv_paper_to_chunks,
)
from chatqec.ingestion.fetchers.url import URLMetadata, markdown_to_chunks
from chatqec.ingestion.registry import (
    InvalidTransitionError,
    RecordNotFoundError,
    Registry,
    SourceRecord,
    SourceStatus,
    SourceType,
)
from chatqec.ingestion.resolver import (
    AmbiguousResolutionError,
    NoMatchError,
    ResolverHit,
    resolve_arxiv_paper,
)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="chatqec")
def main() -> None:
    """ChatQEC — open-source QEC research assistant."""


@main.command()
@click.argument("question", nargs=-1, required=True)
def ask(question: tuple[str, ...]) -> None:
    """Answer a single QEC question (one-shot)."""
    bot = ChatQEC()
    answer = bot.ask(" ".join(question))
    click.echo(answer.text)


@main.group()
@click.option(
    "--registry",
    "registry_path",
    type=click.Path(path_type=Path),
    default=Path("sources/registry.jsonl"),
    show_default=True,
    help="Path to the JSONL source registry.",
)
@click.pass_context
def sources(ctx: click.Context, registry_path: Path) -> None:
    """Inspect and manage the source registry."""
    ctx.obj = Registry(registry_path)


@sources.command("list")
@click.option(
    "--status",
    type=click.Choice([s.value for s in SourceStatus], case_sensitive=False),
    default=None,
    help="Filter by lifecycle status.",
)
@click.pass_obj
def sources_list(reg: Registry, status: str | None) -> None:
    """List registry records as a table."""
    console = Console()
    table = Table(show_header=True, header_style="bold")
    table.add_column("id", no_wrap=True)
    table.add_column("type")
    table.add_column("title")
    table.add_column("tier")
    table.add_column("status")
    table.add_column("year")
    table.add_column("verified")
    status_enum = SourceStatus(status) if status else None
    rows = list(reg.filter(status=status_enum)) if status_enum else list(reg.all())
    for r in rows:
        table.add_row(
            r.id,
            r.type.value,
            r.title[:60],
            r.authority_tier,
            r.status.value,
            str(r.year or ""),
            "✓" if r.verified else "·",
        )
    console.print(table)
    console.print(f"[dim]{len(rows)} records[/]")


@sources.command("show")
@click.argument("record_id")
@click.pass_obj
def sources_show(reg: Registry, record_id: str) -> None:
    """Print one record as pretty JSON."""
    try:
        rec = reg.get(record_id)
    except RecordNotFoundError:
        click.echo(f"record not found: {record_id}", err=True)
        raise SystemExit(2) from None
    click.echo(rec.model_dump_json(indent=2))


@sources.command("mark-retracted")
@click.argument("record_id")
@click.pass_obj
def sources_mark_retracted(reg: Registry, record_id: str) -> None:
    """Manually mark a source as retracted (excluded from retrieval)."""
    try:
        updated = reg.update_status(record_id, SourceStatus.RETRACTED)
    except RecordNotFoundError:
        click.echo(f"record not found: {record_id}", err=True)
        raise SystemExit(2) from None
    except InvalidTransitionError as e:
        click.echo(f"invalid transition: {e}", err=True)
        raise SystemExit(3) from None
    click.echo(f"marked {updated.id} as {updated.status.value}")


@sources.command("compact")
@click.option(
    "--older-than-days",
    type=int,
    default=90,
    show_default=True,
    help="Retracted/superseded records older than this are candidates.",
)
@click.option(
    "--dry-run/--apply",
    default=True,
    help="Dry run (default) prints candidates without removing them.",
)
@click.pass_obj
def sources_compact(reg: Registry, older_than_days: int, dry_run: bool) -> None:
    """List or purge records eligible for compaction.

    NOTE: physical purge of chunks from Qdrant lands in Milestone 5.
    In M2, --apply is rejected with a guard.
    """
    now = datetime.now(tz=UTC)
    candidates = list(reg.compact_candidates(older_than_days=older_than_days, now=now))
    for rec in candidates:
        click.echo(f"{rec.id}\t{rec.status.value}\t{rec.last_checked_at.isoformat()}")
    click.echo(f"\n{len(candidates)} candidates")
    if not dry_run:
        click.echo("--apply not yet supported (wired up in Milestone 5)", err=True)
        raise SystemExit(4)


@main.group()
@click.option(
    "--dir",
    "canonical_dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False, dir_okay=True),
    default=Path("knowledge/canonical"),
    show_default=True,
    help="Path to the canonical wiki directory.",
)
@click.pass_context
def canonical(ctx: click.Context, canonical_dir: Path) -> None:
    """Inspect and validate the canonical wiki layer."""
    ctx.obj = canonical_dir


@canonical.command("list")
@click.pass_obj
def canonical_list(canonical_dir: Path) -> None:
    """List all canonical pages by topic_slug."""
    pages = load_canonical_directory(canonical_dir)
    console = Console()
    table = Table(show_header=True, header_style="bold")
    for col in ("topic_slug", "title", "aliases", "see_also", "last_reviewed"):
        table.add_column(col)
    for p in pages:
        fm = p.frontmatter
        table.add_row(
            fm.topic_slug,
            fm.title,
            ", ".join(fm.aliases),
            ", ".join(fm.see_also),
            fm.last_reviewed.isoformat(),
        )
    console.print(table)
    console.print(f"[dim]{len(pages)} pages[/]")


@canonical.command("validate")
@click.pass_obj
def canonical_validate(canonical_dir: Path) -> None:
    """Validate frontmatter + cross-references for every canonical page."""
    try:
        pages = load_canonical_directory(canonical_dir)
    except CanonicalLoadError as e:
        click.echo(f"validation failed: {e}", err=True)
        raise SystemExit(2) from None
    click.echo(f"ok — {len(pages)} pages parsed and cross-references resolve")


def _fetch_url_as_markdown(url: str) -> tuple[str, str, str]:
    """Default URL fetcher: HTTP GET + trafilatura → markdown. Overridable in tests."""
    import requests
    import trafilatura

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    extracted = trafilatura.extract(
        resp.text, output_format="markdown", include_links=True, with_metadata=True
    )
    if not extracted:
        return (url, "", "")
    md_text = extracted
    # Best-effort metadata extraction
    try:
        meta = trafilatura.extract_metadata(resp.text)
        title = (meta.title if meta and meta.title else url) or url
        publish_date = (meta.date if meta and meta.date else "") or ""
    except Exception:
        title, publish_date = url, ""
    return title, md_text, publish_date


def _resolve_and_fetch_arxiv(
    title_keywords: str,
    authors: list[str],
    year: int | None,
) -> tuple[ResolverHit, str]:
    """Resolve title+author to an arxiv id, then fetch the abstract+full text."""
    import arxiv

    hit = resolve_arxiv_paper(title_keywords=title_keywords, authors=authors, expected_year=year)
    search = arxiv.Search(id_list=[hit.arxiv_id])
    [result] = list(arxiv.Client().results(search))
    # For M4 we ingest the abstract only; full-text PDF parsing lands here
    # behind a flag in M5 once `unstructured` is wired into the pipeline.
    return hit, result.summary


@main.command("ingest")
@click.option(
    "--chunks-dir",
    type=click.Path(path_type=Path),
    default=Path("data/chunks"),
    show_default=True,
    help="Directory to write JSONL chunk files.",
)
@click.option(
    "--registry",
    "registry_path",
    type=click.Path(path_type=Path),
    default=Path("sources/registry.jsonl"),
    show_default=True,
    help="Path to the source registry.",
)
@click.option("--url", "url_value", default=None, help="Ingest a single URL (blog/HTML).")
@click.option(
    "--arxiv",
    "arxiv_id_value",
    default=None,
    help="Ingest by arxiv id. WARNING: §7.4 — prefer --resolve-by-title.",
)
@click.option(
    "--resolve-by-title",
    "title_keywords",
    default=None,
    help="Resolve a paper by title keywords and ingest it.",
)
@click.option(
    "--author",
    "authors_opt",
    multiple=True,
    help="Author last names (use with --resolve-by-title).",
)
@click.option(
    "--year",
    type=int,
    default=None,
    help="Expected publication year (helps disambiguate).",
)
def ingest(
    chunks_dir: Path,
    registry_path: Path,
    url_value: str | None,
    arxiv_id_value: str | None,
    title_keywords: str | None,
    authors_opt: tuple[str, ...],
    year: int | None,
) -> None:
    """Ingest one source. Writes chunks to disk and a pending_review record."""
    sink = ChunkSink(chunks_dir)
    reg = Registry(registry_path)
    now = datetime.now(tz=UTC)

    if url_value:
        title, md_text, publish_date = _fetch_url_as_markdown(url_value)
        md = URLMetadata(url=url_value, title=title, publish_date=publish_date)
        chunks = markdown_to_chunks(md, markdown_text=md_text, target_tokens=512, overlap_tokens=64)
        source_id = chunks[0].source_id
        sink.write_many(source_id, chunks)
        reg.append(
            SourceRecord(
                id=source_id,
                type=SourceType.BLOG,
                url=url_value,
                title=title,
                authority_tier="blog",
                ingested_at=now,
                last_checked_at=now,
                status=SourceStatus.PENDING_REVIEW,
                chunk_count=len(chunks),
                chunk_ids=[c.chunk_id for c in chunks],
            )
        )
        click.echo(f"ingested {source_id}: {len(chunks)} chunks → {chunks_dir}")
        return

    if title_keywords:
        try:
            hit, body = _resolve_and_fetch_arxiv(title_keywords, list(authors_opt), year)
        except (NoMatchError, AmbiguousResolutionError) as e:
            click.echo(f"resolver failed: {e}", err=True)
            raise SystemExit(2) from None
        md_arxiv = ArxivMetadata(
            arxiv_id=hit.arxiv_id,
            title=hit.title,
            authors=[hit.first_author],
            year=hit.year,
            abstract=body,
            pdf_url="",
        )
        chunks = arxiv_paper_to_chunks(
            md_arxiv, body_text=body, target_tokens=512, overlap_tokens=64
        )
        source_id = chunks[0].source_id
        sink.write_many(source_id, chunks)
        reg.append(
            SourceRecord(
                id=source_id,
                type=SourceType.ARXIV,
                url=f"https://arxiv.org/abs/{hit.arxiv_id}",
                title=hit.title,
                authors=[hit.first_author],
                year=hit.year,
                authority_tier="preprint",
                ingested_at=now,
                last_checked_at=now,
                status=SourceStatus.PENDING_REVIEW,
                chunk_count=len(chunks),
                chunk_ids=[c.chunk_id for c in chunks],
                verified=True,
            )
        )
        click.echo(f"ingested {source_id}: {len(chunks)} chunks (resolved from title)")
        return

    if arxiv_id_value:
        click.echo(
            "WARNING: --arxiv <id> trusts the supplied id. "
            "Per spec §7.4 prefer --resolve-by-title for LLM-suggested papers.",
            err=True,
        )
        # M4 implements only the resolve-by-title path live; the --arxiv path
        # delegates to the same fetcher with id_list=[arxiv_id_value].
        # Full implementation lands in M5 when we wire up _fetch_arxiv_by_id.
        click.echo("--arxiv <id> ingestion is not yet supported (lands in M5)", err=True)
        raise SystemExit(3)

    click.echo("nothing to ingest — pass --url, --resolve-by-title, or --arxiv", err=True)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
