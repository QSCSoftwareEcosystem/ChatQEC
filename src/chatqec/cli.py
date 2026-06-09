"""Click-based CLI entry point."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.console import Console
from rich.table import Table

from chatqec import __version__

if TYPE_CHECKING:
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
from chatqec.ingestion.watchlist import Watchlist, load_watchlist


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="chatqec")
def main() -> None:
    """ChatQEC — open-source QEC research assistant."""


def _build_chatqec() -> ChatQEC:  # pragma: no cover
    """Build a ChatQEC instance from on-disk config. Overridable in tests."""
    import os

    import anthropic
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from chatqec.api import ChatQEC
    from chatqec.conversation.classifier import OffTopicClassifier
    from chatqec.conversation.rewriter import QueryRewriter
    from chatqec.conversation.synthesizer import Synthesizer
    from chatqec.retrieval.hybrid import HybridRetriever

    conf_dir = str(Path(os.environ.get("CHATQEC_CONF_DIR", "conf")).resolve())
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="config")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)

    emb_cfg = dict(cfg_dict["embedding"])
    embedder = _make_embedder(emb_cfg)
    sparse = _make_sparse()
    store = _make_store(
        str(cfg_dict["store"]["qdrant_url"]),
        str(cfg_dict["store"]["collection"]),
    )
    reranker = _make_reranker()
    canonical_dir = Path(str(cfg_dict["paths"]["canonical_dir"]))
    canonical_index = _make_canonical_index(canonical_dir)

    retriever = HybridRetriever(
        store=store,  # type: ignore[arg-type]
        embedder=embedder,  # type: ignore[arg-type]
        sparse_encoder=sparse,  # type: ignore[arg-type]
        reranker=reranker,  # type: ignore[arg-type]
        canonical_index=canonical_index,  # type: ignore[arg-type]
    )

    client = anthropic.Anthropic()
    models = cfg_dict["models"]
    return ChatQEC(
        classifier=OffTopicClassifier(client=client, model=str(models["classifier"])),
        rewriter=QueryRewriter(client=client, model=str(models["rewriter"])),
        retriever=retriever,
        synthesizer=Synthesizer(client=client, model=str(models["synthesizer"])),
        canonical_index=canonical_index,  # type: ignore[arg-type]
    )


@main.command()
@click.argument("question", nargs=-1, required=True)
def ask(question: tuple[str, ...]) -> None:
    """Answer a single QEC question (one-shot)."""
    bot = _build_chatqec()
    answer = bot.ask(" ".join(question))
    click.echo(answer.text)
    if answer.citations:
        click.echo("\nSources:")
        for c in answer.citations:
            click.echo(f"  - [{c.source_id}] {c.title}")
    click.echo(f"\n(confidence: {answer.confidence})")


@main.command()
def chat() -> None:
    """Interactive REPL chat. Type /quit to exit, /clear to wipe memory."""
    bot = _build_chatqec()
    click.echo("ChatQEC interactive chat. Type /quit to exit, /clear to reset memory.")
    while True:
        try:
            line = click.prompt("you", prompt_suffix="> ", default="", show_default=False)
        except (click.exceptions.Abort, EOFError):
            click.echo("\nbye.")
            return
        line = line.strip()
        if not line:
            continue
        if line == "/quit":
            click.echo("bye.")
            return
        if line == "/clear":
            bot.memory.clear()
            click.echo("(memory cleared)")
            continue
        answer = bot.ask(line)
        click.echo(f"chatqec> {answer.text}")
        if answer.citations:
            click.echo(f"  sources: {', '.join(c.source_id for c in answer.citations)}")


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

    With --apply, retracted/superseded chunks are physically removed from Qdrant.
    """
    now = datetime.now(tz=UTC)
    candidates = list(reg.compact_candidates(older_than_days=older_than_days, now=now))
    for rec in candidates:
        click.echo(f"{rec.id}\t{rec.status.value}\t{rec.last_checked_at.isoformat()}")
    click.echo(f"\n{len(candidates)} candidates")
    if not dry_run:
        # Deletion-only path: skip embedder/sparse_encoder construction.
        from chatqec.store.qdrant import ChatQecStore, StoreConfig

        store = ChatQecStore(
            StoreConfig(
                qdrant_url="http://localhost:6333",
                collection="chatqec_chunks",
                vector_dim=768,
                on_disk_payload=True,
            )
        )
        purged: list[str] = []
        for rec in candidates:
            store.delete_by_source_id(rec.id)
            purged.append(rec.id)
        click.echo(f"purged {len(purged)} sources from Qdrant")


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


def _collect_watchlist_candidates(wl: Watchlist) -> list[tuple[str, str, list[str], int | None]]:
    """Return a list of (candidate_id, title, authors, year).

    In M4 this stub returns an empty list; M10 will wire up the real
    arxiv/github/youtube/rss polling logic behind a GitHub Action.
    """
    return []


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
        from chatqec.ingestion.fetchers.arxiv import fetch_arxiv_by_id

        md_id = fetch_arxiv_by_id(arxiv_id_value)
        chunks = arxiv_paper_to_chunks(
            md_id, body_text=md_id.abstract, target_tokens=512, overlap_tokens=64
        )
        source_id = chunks[0].source_id
        sink.write_many(source_id, chunks)
        reg.append(
            SourceRecord(
                id=source_id,
                type=SourceType.ARXIV,
                url=f"https://arxiv.org/abs/{md_id.arxiv_id}",
                title=md_id.title,
                authors=md_id.authors,
                year=md_id.year,
                authority_tier="preprint",
                ingested_at=now,
                last_checked_at=now,
                status=SourceStatus.PENDING_REVIEW,
                chunk_count=len(chunks),
                chunk_ids=[c.chunk_id for c in chunks],
                verified=False,
            )
        )
        click.echo(f"ingested {source_id}: {len(chunks)} chunks (TRUSTED id — verify!)")
        return

    click.echo("nothing to ingest — pass --url, --resolve-by-title, or --arxiv", err=True)
    raise SystemExit(1)


@main.group()
@click.option(
    "--watchlist",
    "watchlist_path",
    type=click.Path(path_type=Path),
    default=Path("sources/watchlist.yaml"),
    show_default=True,
)
@click.pass_context
def watch(ctx: click.Context, watchlist_path: Path) -> None:
    """Inspect and refresh the source watchlist."""
    ctx.obj = load_watchlist(watchlist_path)


@watch.command("refresh")
@click.option(
    "--dry-run/--apply",
    default=True,
    help="Dry run lists candidates; --apply not yet supported (lands in M10).",
)
@click.pass_obj
def watch_refresh(wl: Watchlist, dry_run: bool) -> None:
    """Pull fresh candidates from each watchlist entry."""
    candidates = _collect_watchlist_candidates(wl)
    if not dry_run:
        click.echo("--apply not yet supported (lands in M10 with GitHub Action)", err=True)
        raise SystemExit(4)
    for cid, title, authors, year in candidates:
        click.echo(f"{cid}\t{title}\t{', '.join(authors)}\t{year}")
    click.echo(f"\n{len(candidates)} candidates")


def _make_embedder(cfg: dict[str, object]) -> object:
    """Build the configured embedder. Overridable for tests."""
    from chatqec.embedding.base import EmbedderConfig

    if cfg["provider"] == "local":
        from chatqec.embedding.local import LocalEmbedder

        return LocalEmbedder(EmbedderConfig(**cfg))  # type: ignore[arg-type]
    from chatqec.embedding.gemini import GeminiEmbedder

    return GeminiEmbedder(EmbedderConfig(**cfg))  # type: ignore[arg-type]


def _make_sparse() -> object:
    from chatqec.embedding.sparse import BM25Encoder

    return BM25Encoder()


def _make_store(url: str, collection: str) -> object:
    from chatqec.store.qdrant import ChatQecStore, StoreConfig

    return ChatQecStore(
        StoreConfig(qdrant_url=url, collection=collection, vector_dim=768, on_disk_payload=True)
    )


def _make_reranker() -> object:
    from chatqec.retrieval.reranker import CrossEncoderReranker

    return CrossEncoderReranker()


def _make_canonical_index(canonical_dir: Path) -> object:
    from chatqec.canonical.loader import load_canonical_directory
    from chatqec.retrieval.canonical_injector import CanonicalIndex

    pages = load_canonical_directory(canonical_dir) if canonical_dir.exists() else []
    return CanonicalIndex(pages=pages)


@main.command("embed")
@click.option(
    "--chunks-dir",
    type=click.Path(path_type=Path),
    default=Path("data/chunks"),
    show_default=True,
)
@click.option(
    "--registry",
    "registry_path",
    type=click.Path(path_type=Path),
    default=Path("sources/registry.jsonl"),
    show_default=True,
)
@click.option("--source-id", default=None, help="Embed a single source by id.")
@click.option(
    "--all-pending",
    "all_pending",
    is_flag=True,
    default=False,
    help="Embed every pending_review source in the registry.",
)
@click.option("--qdrant-url", default="http://localhost:6333", show_default=True)
@click.option("--collection", default="chatqec_chunks", show_default=True)
@click.option(
    "--provider",
    default="gemini",
    show_default=True,
    type=click.Choice(["gemini", "local"]),
)
def embed(
    chunks_dir: Path,
    registry_path: Path,
    source_id: str | None,
    all_pending: bool,
    qdrant_url: str,
    collection: str,
    provider: str,
) -> None:
    """Embed chunk files into Qdrant and promote pending_review → active."""
    from chatqec.ingestion.pipeline import EmbeddingPipeline

    if not source_id and not all_pending:
        click.echo("pass --source-id <id> or --all-pending", err=True)
        raise SystemExit(1)

    emb_cfg = {
        "provider": provider,
        "model": "text-embedding-004",
        "batch_size": 96,
        "local_model": "nomic-ai/nomic-embed-text-v1.5",
    }
    pipe = EmbeddingPipeline(
        chunks_dir=chunks_dir,
        registry=Registry(registry_path),
        embedder=_make_embedder(emb_cfg),  # type: ignore[arg-type]
        sparse_encoder=_make_sparse(),  # type: ignore[arg-type]
        store=_make_store(qdrant_url, collection),  # type: ignore[arg-type]
    )
    if source_id:
        result = pipe.embed_source(source_id)
        click.echo(f"embedded {result.source_id}: {result.chunks_embedded} chunks")
    else:
        results = pipe.embed_all_pending()
        total = sum(r.chunks_embedded for r in results)
        click.echo(f"{len(results)} sources, {total} chunks embedded")


@main.command("retrieve")
@click.option("--query", "query_str", required=True, help="Single query string.")
@click.option("--top-k", type=int, default=8, show_default=True)
@click.option("--candidate-k", type=int, default=50, show_default=True)
@click.option("--qdrant-url", default="http://localhost:6333", show_default=True)
@click.option("--collection", default="chatqec_chunks", show_default=True)
@click.option(
    "--canonical-dir",
    type=click.Path(path_type=Path),
    default=Path("knowledge/canonical"),
    show_default=True,
)
@click.option(
    "--provider", default="gemini", show_default=True, type=click.Choice(["gemini", "local"])
)
def retrieve(
    query_str: str,
    top_k: int,
    candidate_k: int,
    qdrant_url: str,
    collection: str,
    canonical_dir: Path,
    provider: str,
) -> None:
    """Debug command: run the hybrid retriever for one query and print results."""
    from chatqec.retrieval.hybrid import HybridRetriever

    emb_cfg: dict[str, object] = {
        "provider": provider,
        "model": "text-embedding-004",
        "batch_size": 96,
        "local_model": "nomic-ai/nomic-embed-text-v1.5",
    }
    rr = HybridRetriever(
        store=_make_store(qdrant_url, collection),  # type: ignore[arg-type]
        embedder=_make_embedder(emb_cfg),  # type: ignore[arg-type]
        sparse_encoder=_make_sparse(),  # type: ignore[arg-type]
        reranker=_make_reranker(),  # type: ignore[arg-type]
        canonical_index=_make_canonical_index(canonical_dir),  # type: ignore[arg-type]
    )
    result = rr.retrieve(queries=[query_str], top_k=top_k, candidate_k=candidate_k)
    console = Console()
    table = Table(show_header=True, header_style="bold")
    for col in ("rank", "chunk_id", "source_id", "tier", "score", "title"):
        table.add_column(col)
    for i, c in enumerate(result.chunks):
        table.add_row(
            str(i + 1), c.chunk_id, c.source_id, c.authority_tier, f"{c.score:.3f}", c.title[:60]
        )
    console.print(table)
    if result.matched_topics:
        console.print(f"[dim]Matched canonical topics: {', '.join(result.matched_topics)}[/]")


if __name__ == "__main__":
    main()
