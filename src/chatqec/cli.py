"""Click-based CLI entry point."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from chatqec import __version__

if TYPE_CHECKING:
    from chatqec.api import ChatQEC
    from chatqec.api_types import ImageAttachment
from chatqec.canonical.loader import (
    CanonicalLoadError,
    load_canonical_directory,
)
from chatqec.ingestion.chunk import Chunk, ChunkSink
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
    load_dotenv()


_HF_MODEL_REMAP: dict[str, str] = {  # pragma: no cover
    "claude-haiku-4-5": "meta-llama/Llama-3.1-8B-Instruct",
    "claude-sonnet-4-6": "meta-llama/Llama-3.3-70B-Instruct",
}

_GEMINI_MODEL_REMAP: dict[str, str] = {  # pragma: no cover
    "claude-haiku-4-5": "gemini-2.5-flash",
    "claude-sonnet-4-6": "gemini-2.5-pro",
}

# Vision-capable HF model used when a request carries image content.
_HF_VISION_MODEL = "meta-llama/Llama-4-Scout-17B-16E-Instruct"


def _try_anthropic() -> object | None:  # pragma: no cover
    """Construct an Anthropic client if a key is available; None otherwise."""
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic

        client: object = anthropic.Anthropic()
        return client
    except Exception:
        return None


def _try_huggingface() -> object | None:  # pragma: no cover
    """Construct a HuggingFace client if a token is available; None otherwise."""
    import os

    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")):
        return None
    try:
        from chatqec.providers.huggingface import HuggingFaceLLM

        return HuggingFaceLLM()
    except Exception:
        return None


def _try_gemini() -> object | None:  # pragma: no cover
    """Construct a Gemini client if an API key is available; None otherwise."""
    import os

    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        return None
    try:
        from chatqec.providers.gemini import GeminiLLM

        return GeminiLLM()
    except Exception:
        return None


def _build_llm_client(provider: str) -> object:  # pragma: no cover
    """Construct the LLM client backing classifier+rewriter+synthesizer."""
    from chatqec.providers.fallback import Backend, FallbackLLMClient

    if provider == "huggingface":
        hf = _try_huggingface()
        if hf is None:
            raise RuntimeError("HF_TOKEN (or HUGGINGFACE_API_KEY) is not set")
        return hf
    if provider == "anthropic":
        ant = _try_anthropic()
        if ant is None:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        return ant
    if provider == "gemini":
        gem = _try_gemini()
        if gem is None:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")
        return gem
    if provider == "auto":
        backends: list[Backend] = []
        ant = _try_anthropic()
        if ant is not None:
            backends.append(Backend(name="anthropic", client=ant))
        gem = _try_gemini()
        if gem is not None:
            backends.append(
                Backend(
                    name="gemini",
                    client=gem,
                    model_remap=_GEMINI_MODEL_REMAP,
                    default_model="gemini-2.5-pro",
                )
            )
        hf = _try_huggingface()
        if hf is not None:
            backends.append(
                Backend(
                    name="huggingface",
                    client=hf,
                    model_remap=_HF_MODEL_REMAP,
                    default_model="meta-llama/Llama-3.3-70B-Instruct",
                    vision_model=_HF_VISION_MODEL,
                )
            )
        if not backends:
            raise RuntimeError(
                "no LLM provider configured; set ANTHROPIC_API_KEY, GEMINI_API_KEY, and/or HF_TOKEN"
            )
        return FallbackLLMClient(backends=backends)
    raise ValueError(f"unknown LLM provider: {provider!r}")


def _role_models(
    models_cfg: dict[str, object], provider: str
) -> dict[str, str]:  # pragma: no cover
    """Resolve per-role model IDs for the selected provider."""
    if provider == "huggingface":
        hf = models_cfg.get("hf") or {}
        assert isinstance(hf, dict), "models.hf must be a mapping"
        return {
            "classifier": str(hf.get("classifier", "meta-llama/Llama-3.1-8B-Instruct")),
            "rewriter": str(hf.get("rewriter", "meta-llama/Llama-3.1-8B-Instruct")),
            "synthesizer": str(hf.get("synthesizer", "meta-llama/Llama-3.3-70B-Instruct")),
        }
    if provider == "gemini":
        gem = models_cfg.get("gemini") or {}
        assert isinstance(gem, dict), "models.gemini must be a mapping"
        return {
            "classifier": str(gem.get("classifier", "gemini-2.5-flash")),
            "rewriter": str(gem.get("rewriter", "gemini-2.5-flash")),
            "synthesizer": str(gem.get("synthesizer", "gemini-2.5-pro")),
        }
    return {
        "classifier": str(models_cfg["classifier"]),
        "rewriter": str(models_cfg["rewriter"]),
        "synthesizer": str(models_cfg["synthesizer"]),
    }


def _build_chatqec(mcp_server: str | None = None) -> ChatQEC:  # pragma: no cover
    """Build a ChatQEC instance from on-disk config. Overridable in tests.

    When ``mcp_server`` is set, the synthesizer is given an MCP adapter wired to
    a stdio-spawned MCP server (e.g. ``chatqec-mcp-tools``).
    """
    import os

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
    reranker = _make_reranker(str(cfg_dict["models"].get("reranker") or "") or None)
    canonical_dir = Path(str(cfg_dict["paths"]["canonical_dir"]))
    canonical_index = _make_canonical_index(canonical_dir)

    retriever = HybridRetriever(
        store=store,  # type: ignore[arg-type]
        embedder=embedder,  # type: ignore[arg-type]
        sparse_encoder=sparse,  # type: ignore[arg-type]
        reranker=reranker,  # type: ignore[arg-type]
        canonical_index=canonical_index,  # type: ignore[arg-type]
    )

    models = cfg_dict["models"]
    provider = str(models.get("provider", "anthropic"))
    client = _build_llm_client(provider)
    role_models = _role_models(models, provider)

    synthesizer = Synthesizer(client=client, model=role_models["synthesizer"])
    if mcp_server:
        if provider not in ("anthropic", "gemini", "auto"):
            raise ValueError(
                "--mcp-server requires the anthropic, gemini, or auto provider; "
                f"got {provider!r} (huggingface has no tool-use support)"
            )
        from chatqec.conversation.mcp_adapter import MCPAdapter
        from chatqec.conversation.mcp_stdio import StdioMCPClient

        synthesizer.mcp_adapter = MCPAdapter(client=StdioMCPClient(server_command=mcp_server))

    return ChatQEC(
        classifier=OffTopicClassifier(client=client, model=role_models["classifier"]),
        rewriter=QueryRewriter(client=client, model=role_models["rewriter"]),
        retriever=retriever,
        synthesizer=synthesizer,
        canonical_index=canonical_index,  # type: ignore[arg-type]
    )


def _start_file_trace(verbose: bool, log_file: str | None) -> None:  # pragma: no cover
    """Enable step tracing to a log file (keeps the terminal clean).

    Prints a single pointer line to stderr so the answer on stdout stays
    pipe-friendly. With no ``--log-file`` a timestamped file under ``logs/`` is
    used.
    """
    if not verbose:
        return
    from chatqec.trace import enable_trace

    path = log_file or f"logs/chatqec-trace-{datetime.now():%Y%m%d-%H%M%S}.log"
    resolved = enable_trace(True, log_file=path, to_stderr=False)
    click.echo(f"📝 verbose trace → {resolved}", err=True)


_VERBOSE_HELP = (
    "Trace every pipeline step (classify, rewrite, retrieve, synthesize, "
    "model/provider, MCP tool calls) to a log file instead of the terminal."
)
_LOG_FILE_HELP = "Path for the verbose trace log (default: logs/chatqec-trace-<timestamp>.log)."

_IMG_EXT_TO_MEDIA = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def _read_image_files(paths: tuple[str, ...]) -> list[ImageAttachment]:
    """Load + validate CLI --image paths into ImageAttachments (≤3, guardrailed)."""
    from chatqec.ingestion.images import prepare_images

    raw: list[tuple[bytes, str, str]] = []
    for p in paths:
        path = Path(p)
        media = _IMG_EXT_TO_MEDIA.get(path.suffix.lower(), "image/png")
        raw.append((path.read_bytes(), media, path.name))
    return prepare_images(raw)


@main.command()
@click.argument("question", nargs=-1, required=True)
@click.option(
    "--mcp-server",
    default=None,
    help="Command to launch an MCP tool server (e.g. 'chatqec-mcp-tools').",
)
@click.option("-v", "--verbose", is_flag=True, default=False, help=_VERBOSE_HELP)
@click.option("--log-file", default=None, help=_LOG_FILE_HELP)
@click.option(
    "--image",
    "image_paths",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Attach an image (repeatable, ≤3) — asks about the figure.",
)
def ask(
    question: tuple[str, ...],
    mcp_server: str | None,
    verbose: bool,
    log_file: str | None,
    image_paths: tuple[str, ...],
) -> None:
    """Answer a single QEC question (one-shot)."""
    from chatqec.trace import disable_trace

    _start_file_trace(verbose, log_file)
    bot = _build_chatqec(mcp_server=mcp_server)
    answer = bot.ask(" ".join(question), images=_read_image_files(image_paths))
    disable_trace()
    click.echo(answer.text)
    if answer.tool_calls:
        click.echo("\nTool calls:")
        for t in answer.tool_calls:
            click.echo(f"  - {t.name}({t.input})")
    if answer.citations:
        click.echo("\nSources:")
        for c in answer.citations:
            click.echo(f"  - [{c.source_id}] {c.title}")
            if c.url:
                click.echo(f"    {c.url}")
    click.echo(f"\n(confidence: {answer.confidence})")


@main.command()
@click.option(
    "--mcp-server",
    default=None,
    help="Command to launch an MCP tool server (e.g. 'chatqec-mcp-tools').",
)
@click.option("-v", "--verbose", is_flag=True, default=False, help=_VERBOSE_HELP)
@click.option("--log-file", default=None, help=_LOG_FILE_HELP)
def chat(mcp_server: str | None, verbose: bool, log_file: str | None) -> None:
    """Interactive REPL chat. Type /quit to exit, /clear to wipe memory."""
    from chatqec.trace import disable_trace

    _start_file_trace(verbose, log_file)
    bot = _build_chatqec(mcp_server=mcp_server)
    click.echo("ChatQEC interactive chat. Type /quit to exit, /clear to reset memory.")
    while True:
        try:
            line = click.prompt("you", prompt_suffix="> ", default="", show_default=False)
        except (click.exceptions.Abort, EOFError):
            click.echo("\nbye.")
            disable_trace()
            return
        line = line.strip()
        if not line:
            continue
        if line == "/quit":
            click.echo("bye.")
            disable_trace()
            return
        if line == "/clear":
            bot.memory.clear()
            click.echo("(memory cleared)")
            continue
        answer = bot.ask(line)
        click.echo(f"chatqec> {answer.text}")
        if answer.citations:
            click.echo("  sources:")
            for c in answer.citations:
                click.echo(f"    - [{c.source_id}] {c.title}")
                if c.url:
                    click.echo(f"      {c.url}")


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


@canonical.command("embed")
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
@click.option(
    "--stage-only",
    is_flag=True,
    default=False,
    help="Write chunk files + register sources, but skip embedding into Qdrant.",
)
@click.option("--qdrant-url", default="http://localhost:6333", show_default=True)
@click.option("--collection", default="chatqec_chunks", show_default=True)
@click.option(
    "--provider",
    default="gemini",
    show_default=True,
    type=click.Choice(["gemini", "local"]),
)
@click.pass_obj
def canonical_embed(
    canonical_dir: Path,
    chunks_dir: Path,
    registry_path: Path,
    stage_only: bool,
    qdrant_url: str,
    collection: str,
    provider: str,
) -> None:
    """Chunk canonical pages, register them, and embed into Qdrant for hybrid retrieval."""
    from datetime import UTC, datetime

    from chatqec.canonical.embed import stage_canonical_directory
    from chatqec.ingestion.pipeline import EmbeddingPipeline

    registry = Registry(registry_path)
    staged = stage_canonical_directory(
        canonical_dir,
        chunks_dir=chunks_dir,
        registry=registry,
        now=datetime.now(tz=UTC),
    )
    click.echo(f"staged {len(staged)} canonical pages → {chunks_dir}")
    if stage_only:
        click.echo("stage-only: skipped embedding (run `chatqec embed --all-pending` to vectorize)")
        return

    emb_cfg = {
        "provider": provider,
        "model": "gemini-embedding-001",
        "batch_size": 96,
        "local_model": "nomic-ai/nomic-embed-text-v1.5",
    }
    pipe = EmbeddingPipeline(
        chunks_dir=chunks_dir,
        registry=registry,
        embedder=_make_embedder(emb_cfg),  # type: ignore[arg-type]
        sparse_encoder=_make_sparse(),  # type: ignore[arg-type]
        store=_make_store(qdrant_url, collection),  # type: ignore[arg-type]
    )
    total = 0
    for source_id in staged:
        result = pipe.embed_source(source_id)
        total += result.chunks_embedded
    click.echo(f"embedded {len(staged)} pages, {total} chunks → {collection}")


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


def _arxiv_id_to_chunks(
    arxiv_id_value: str, full_text: bool
) -> tuple[ArxivMetadata, list[Chunk]]:  # pragma: no cover - network + subprocess
    """Resolve an arXiv id to (metadata, chunks).

    Default: abstract only. With ``full_text``: LaTeX source, then an isolated
    Marker PDF extraction, then abstract — whichever succeeds first.
    """
    from chatqec.ingestion.fetchers.arxiv import fetch_arxiv_by_id

    def _abstract() -> tuple[ArxivMetadata, list[Chunk]]:
        md = fetch_arxiv_by_id(arxiv_id_value)
        chunks = arxiv_paper_to_chunks(
            md, body_text=md.abstract, target_tokens=512, overlap_tokens=64
        )
        return md, chunks

    if not full_text:
        return _abstract()

    from chatqec.ingestion.fetchers import arxiv_source
    from chatqec.ingestion.fetchers.pdf_marker import MarkerUnavailableError

    try:
        md_id, chunks = arxiv_source.fetch_arxiv_fulltext(
            arxiv_id_value, target_tokens=512, overlap_tokens=64
        )
        click.echo(f"  full-text LaTeX source → {len(chunks)} chunks", err=True)
        return md_id, chunks
    except arxiv_source.SourceUnavailableError:
        click.echo("  no LaTeX source — extracting the PDF with Marker (isolated)...", err=True)

    try:
        md_id, chunks = arxiv_source.fetch_arxiv_pdf_fulltext(
            arxiv_id_value, target_tokens=512, overlap_tokens=64
        )
        click.echo(f"  Marker PDF extraction → {len(chunks)} chunks", err=True)
        return md_id, chunks
    except (MarkerUnavailableError, OSError) as e:
        click.echo(f"  Marker unavailable ({e}) — falling back to abstract only.", err=True)
        return _abstract()


def _collect_watchlist_candidates(wl: Watchlist) -> list[tuple[str, str, list[str], int | None]]:
    """Return a list of (candidate_id, title, authors, year).

    Backed by ``chatqec.ingestion.watchlist_refresh.refresh_all`` (arXiv + GitHub).
    """
    from chatqec.ingestion.watchlist_refresh import refresh_all

    now = datetime.now(tz=UTC)
    return [(c.source_id, c.title, c.authors, c.year) for c in refresh_all(wl, now=now)]


def _marker_page_range(pages: str | None) -> tuple[str | None, str]:
    """Convert a 1-based printed page range to Marker's 0-based range + id suffix.

    ``"425-499"`` → (``"424-498"``, ``":pp425-499"``). Returns ``(None, "")`` for
    a whole-file ingest.
    """
    if not pages:
        return None, ""
    start_str, _, end_str = pages.partition("-")
    start = int(start_str)
    end = int(end_str) if end_str else start
    return f"{start - 1}-{end - 1}", f":pp{pages}"


def _ingest_pdf(  # pragma: no cover - drives Marker subprocess + disk I/O
    pdf_path: Path,
    *,
    sink: ChunkSink,
    reg: Registry,
    now: datetime,
    pages: str | None,
    title: str | None,
    author: str | None,
    year: int | None,
    tier: str,
    license_str: str | None,
    chunks_dir: Path,
    describe_figures: bool = True,
) -> None:
    from chatqec.ingestion.fetchers.pdf import build_pdf_metadata, text_to_chunks
    from chatqec.ingestion.fetchers.pdf_marker import run_marker_on_pdf
    from chatqec.ingestion.figures import annotate_figures
    from chatqec.ingestion.ids import pdf_id

    page_range, suffix = _marker_page_range(pages)
    md = build_pdf_metadata(
        pdf_path,
        title=title or pdf_path.stem,
        author=author or "",
        year=year or 0,
    )
    source_id = pdf_id(md.content_sha256) + suffix
    images_dir = Path("data/figures") / source_id.replace(":", "_").replace("/", "_")
    click.echo(f"extracting {pdf_path.name} (pages={pages or 'all'}) via Marker…", err=True)
    body = run_marker_on_pdf(pdf_path, page_range=page_range, images_dir=images_dir)
    if describe_figures:
        from chatqec.ingestion.figure_vision import LlamaVisionDescriber

        click.echo("describing figures…", err=True)
        body, _figs = annotate_figures(body, describer=LlamaVisionDescriber())
    chunks = text_to_chunks(md, body_text=body, tier=tier, source_id_suffix=suffix)
    if not chunks:
        click.echo(f"no chunks produced from {pdf_path} — extraction empty", err=True)
        raise SystemExit(1)
    source_id = chunks[0].source_id
    sink.write_many(source_id, chunks)
    reg.append(
        SourceRecord(
            id=source_id,
            type=SourceType.PDF,
            url="",
            title=md.title,
            authors=[author] if author else [],
            year=year,
            content_sha256=md.content_sha256,
            license=license_str,
            authority_tier=tier,
            ingested_at=now,
            last_checked_at=now,
            status=SourceStatus.PENDING_REVIEW,
            chunk_count=len(chunks),
            chunk_ids=[c.chunk_id for c in chunks],
            verified=True,
            notes=f"pdf={pdf_path.name} pages={pages or 'all'}",
        )
    )
    click.echo(f"ingested {source_id}: {len(chunks)} chunks → {chunks_dir}")


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
@click.option(
    "--full-text",
    "full_text",
    is_flag=True,
    default=False,
    help="Ingest the full LaTeX source (with --arxiv) instead of the abstract only.",
)
@click.option(
    "--pdf",
    "pdf_path_value",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="Ingest a local PDF (e.g. a book chapter) via Marker.",
)
@click.option(
    "--pages",
    "pages_value",
    default=None,
    help="1-based PDF page range for --pdf, e.g. '425-499' (a single chapter).",
)
@click.option("--title", "title_value", default=None, help="Title for --pdf source.")
@click.option("--pdf-author", "pdf_author_value", default=None, help="Author(s) for --pdf source.")
@click.option(
    "--tier",
    "tier_value",
    default="peer_reviewed",
    show_default=True,
    help="Authority tier for --pdf (canonical|peer_reviewed|preprint|blog|video).",
)
@click.option(
    "--source-license", "license_value", default=None, help="License/notes for --pdf provenance."
)
@click.option(
    "--describe-figures/--no-describe-figures",
    default=True,
    help="Run the local vision model to describe extracted figures.",
)
def ingest(
    chunks_dir: Path,
    registry_path: Path,
    url_value: str | None,
    arxiv_id_value: str | None,
    title_keywords: str | None,
    authors_opt: tuple[str, ...],
    year: int | None,
    full_text: bool,
    pdf_path_value: Path | None,
    pages_value: str | None,
    title_value: str | None,
    pdf_author_value: str | None,
    tier_value: str,
    license_value: str | None,
    describe_figures: bool,
) -> None:
    """Ingest one source. Writes chunks to disk and a pending_review record."""
    sink = ChunkSink(chunks_dir)
    reg = Registry(registry_path)
    now = datetime.now(tz=UTC)

    if pdf_path_value:
        _ingest_pdf(
            pdf_path_value,
            sink=sink,
            reg=reg,
            now=now,
            pages=pages_value,
            title=title_value,
            author=pdf_author_value,
            year=year,
            tier=tier_value,
            license_str=license_value,
            chunks_dir=chunks_dir,
            describe_figures=describe_figures,
        )
        return

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
        md_id, chunks = _arxiv_id_to_chunks(arxiv_id_value, full_text)
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
    help="Dry run (default) lists candidates; --apply appends pending_review records to the registry.",
)
@click.option(
    "--registry",
    "registry_path",
    type=click.Path(path_type=Path),
    default=Path("sources/registry.jsonl"),
    show_default=True,
)
@click.pass_obj
def watch_refresh(wl: Watchlist, dry_run: bool, registry_path: Path) -> None:
    """Pull fresh candidates from each watchlist entry."""
    candidates = _collect_watchlist_candidates(wl)
    for cid, title, authors, year in candidates:
        click.echo(f"{cid}\t{title}\t{', '.join(authors)}\t{year}")
    click.echo(f"\n{len(candidates)} candidates")
    if dry_run:
        return
    reg = Registry(registry_path)
    now = datetime.now(tz=UTC)
    appended = 0
    for cid, title, authors, year in candidates:
        try:
            reg.get(cid)
            continue
        except RecordNotFoundError:
            pass
        try:
            stype = SourceType(cid.split(":", 1)[0])
        except ValueError:
            continue
        try:
            reg.append(
                SourceRecord(
                    id=cid,
                    type=stype,
                    url="",
                    title=title,
                    authors=authors,
                    year=year,
                    authority_tier="preprint" if stype is SourceType.ARXIV else "blog",
                    ingested_at=now,
                    last_checked_at=now,
                    status=SourceStatus.PENDING_REVIEW,
                )
            )
            appended += 1
        except (ValueError, InvalidTransitionError) as e:
            click.echo(f"skip {cid}: {e}", err=True)
    click.echo(f"appended {appended} pending_review records to {registry_path}")


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


def _make_reranker(model_name: str | None = None) -> object:
    from chatqec.retrieval.reranker import CrossEncoderReranker

    if model_name:
        return CrossEncoderReranker(model_name=model_name)
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
        "model": "gemini-embedding-001",
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
        "model": "gemini-embedding-001",
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


def _make_judge_client() -> object:  # pragma: no cover
    """Build the LLM-judge client used in eval runs.

    Reuses the ``auto`` fallback chain so the judge fails over from Anthropic to
    Gemini/HF on credit/auth errors — same robustness as the answer path. The
    judge model (e.g. ``claude-haiku-4-5``) is remapped per backend by the
    fallback client.
    """
    return _build_llm_client("auto")


@main.group("eval")
def eval_group() -> None:
    """Run the eval set and emit metrics."""


@eval_group.command("run")
@click.option(
    "--cases",
    "cases_path",
    type=click.Path(path_type=Path),
    default=Path("eval/qa_set.jsonl"),
    show_default=True,
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=Path("eval/results.json"),
    show_default=True,
)
@click.option("--judge-model", default="claude-haiku-4-5", show_default=True)
def eval_run(cases_path: Path, output_path: Path, judge_model: str) -> None:
    """Run the eval set against the configured ChatQEC and emit JSON metrics."""
    import json

    from chatqec.eval.dataset import load_eval_set
    from chatqec.eval.runner import run_eval

    cases = load_eval_set(cases_path)
    bot = _build_chatqec()
    judge = _make_judge_client()

    report = run_eval(cases=cases, bot=bot, judge_client=judge, judge_model=judge_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "total": report.total,
                "overall_recall_at_k": report.overall_recall_at_k,
                "overall_citation_accuracy": report.overall_citation_accuracy,
                "overall_faithfulness": report.overall_faithfulness,
                "per_subtype": {
                    k: {
                        "n": v.n,
                        "recall_at_k": v.recall_at_k,
                        "citation_accuracy": v.citation_accuracy,
                        "faithfulness": v.faithfulness,
                    }
                    for k, v in report.per_subtype.items()
                },
                "raw": report.raw,
            },
            indent=2,
        )
    )
    click.echo(
        f"eval done: {report.total} cases, "
        f"recall@k={report.overall_recall_at_k:.2f}, "
        f"citation_acc={report.overall_citation_accuracy:.2f}, "
        f"faithfulness={report.overall_faithfulness:.2f}"
    )
    click.echo(f"report: {output_path}")


@main.command("corpus-stats")
@click.option(
    "--chunks-dir",
    type=click.Path(path_type=Path),
    default=Path("data/chunks"),
    show_default=True,
    help="Directory of JSONL chunk files to summarize.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=Path("data/corpus_stats.json"),
    show_default=True,
    help="Where to write the full JSON stats report.",
)
def corpus_stats(chunks_dir: Path, output_path: Path) -> None:
    """Summarize the ingested corpus (chunk/token statistics) for reporting."""
    import json

    from chatqec.stats import compute_corpus_stats, load_chunks

    chunks = load_chunks(chunks_dir)
    stats = compute_corpus_stats(chunks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    tok = stats["tokens"]
    click.echo(f"sources={stats['n_sources']}  chunks={stats['n_chunks']}")
    click.echo(
        f"tokens/chunk (~chars/4): total={tok['total']} min={tok['min']} "
        f"max={tok['max']} mean={tok['mean']} median={tok['median']} p95={tok['p95']}"
    )
    cps = stats["chunks_per_source"]
    click.echo(f"chunks/source: min={cps['min']} max={cps['max']} mean={cps['mean']}")
    lc = stats["largest_chunk"]
    if lc:
        click.echo(f"largest chunk: {lc['tokens']} tok  [{lc['source_id']}] {lc['chunk_id']}")
    click.echo(f"by source_type: {stats['by_source_type']}")
    click.echo(f"by tier: {stats['by_tier']}")
    click.echo(f"full report → {output_path}")


if __name__ == "__main__":
    main()
