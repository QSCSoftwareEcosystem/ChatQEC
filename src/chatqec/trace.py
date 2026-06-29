"""Lightweight, switchable execution tracing for ChatQEC.

A single process-wide tracer that emits one record per pipeline step
(classifier, rewriter, retrieval, synthesis, provider selection, tool/MCP
calls) with full payloads. It is **disabled by default** and costs a single
boolean check when off, so ``trace(...)`` calls can be left in hot paths.

Enable it via the CLI ``-v/--verbose`` flag or the Streamlit sidebar toggle.
Records can be routed to:
    - a **log file** (``log_file=``) — the CLI default, so the terminal stays
      clean and shows only the answer; the file gets timestamped, numbered,
      blank-line-separated records,
    - stderr (``to_stderr=True``),
    - and/or an in-memory ``sink`` callable — the Streamlit app uses this to
      capture the trace and render it in the UI.

Design notes:
    - State is module-global on purpose. Threading a tracer object through every
      constructor (classifier, retriever, synthesizer, tool loop, MCP client)
      would touch far more surface than the feature warrants; a global switch
      keeps call sites to ``trace("step", key=value)``.
    - Payloads are NOT truncated by default (``max_len=None``) so the trace is a
      faithful dump of what each step saw/produced. Callers may pass ``max_len``
      for individual noisy fields.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO, Any


def _stderr_sink(line: str) -> None:
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


def _make_file_sink(handle: IO[str]) -> Callable[[str], None]:
    """Build a sink that writes numbered, timestamped, separated records.

    Each record gets a ``[NNN] HH:MM:SS.mmm  step`` header line (added here so
    the in-memory/stderr formats stay clean) followed by its field lines and a
    blank-line separator for scannability.
    """
    counter = {"n": 0}

    def sink(record: str) -> None:
        counter["n"] += 1
        stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        header, _, body = record.partition("\n")
        # ``header`` looks like "[trace] step"; drop the prefix in the file since
        # the line is already unambiguous with the index + timestamp.
        step = header.removeprefix("[trace] ")
        handle.write(f"\n[{counter['n']:03d}] {stamp}  {step}\n")
        if body:
            handle.write(body + "\n")
        handle.flush()

    return sink


@dataclass
class _TracerState:
    """Process-wide tracer state, mutated in place (no ``global`` rebinding)."""

    enabled: bool = False
    sinks: list[Callable[[str], None]] = field(default_factory=list)
    file_handle: IO[str] | None = None
    log_path: Path | None = None


_state = _TracerState()


def _close_file() -> None:
    if _state.file_handle is not None:
        try:
            _state.file_handle.close()
        finally:
            _state.file_handle = None
            _state.log_path = None


def enable_trace(
    enabled: bool = True,
    *,
    sink: Callable[[str], None] | None = None,
    to_stderr: bool = True,
    log_file: str | Path | None = None,
) -> Path | None:
    """Turn tracing on (or off when ``enabled`` is False).

    Args:
        enabled: Master switch. When False this is equivalent to
            :func:`disable_trace`.
        sink: Optional extra destination for each formatted record (e.g. a
            ``list.append`` so the Streamlit app can show the trace).
        to_stderr: Whether to write records to stderr. Ignored when a
            ``log_file`` is given (file logging implies a quiet terminal).
        log_file: Optional path to append a readable, timestamped trace to. Its
            parent directory is created if missing.

    Returns:
        The resolved log-file :class:`~pathlib.Path` when ``log_file`` was given
        and tracing is enabled, otherwise ``None``.
    """
    _close_file()
    _state.enabled = enabled
    _state.sinks = []
    if not enabled:
        return None

    resolved: Path | None = None
    if log_file is not None:
        resolved = Path(log_file).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        handle = resolved.open("a", encoding="utf-8")
        handle.write(
            f"\n===== ChatQEC trace @ {datetime.now().isoformat(timespec='seconds')} =====\n"
        )
        handle.flush()
        _state.file_handle = handle
        _state.log_path = resolved
        _state.sinks.append(_make_file_sink(handle))
    elif to_stderr:
        _state.sinks.append(_stderr_sink)

    if sink is not None:
        _state.sinks.append(sink)
    return resolved


def disable_trace() -> None:
    """Turn tracing off, flush/close any log file, and drop all sinks."""
    _close_file()
    _state.enabled = False
    _state.sinks = []


def is_enabled() -> bool:
    """Return whether tracing is currently active."""
    return _state.enabled


def current_log_path() -> Path | None:
    """Return the active log-file path, if tracing is writing to one."""
    return _state.log_path


def _fmt_value(value: Any, *, max_len: int | None) -> str:
    text = value if isinstance(value, str) else repr(value)
    if max_len is not None and len(text) > max_len:
        text = text[:max_len] + f"… (+{len(text) - max_len} more chars)"
    if "\n" in text:
        # Indent multi-line payloads so they stay visually nested under the key.
        return "\n        " + text.replace("\n", "\n        ")
    return text


def trace(step: str, *, max_len: int | None = None, **fields: Any) -> None:
    """Emit one trace record for ``step`` with arbitrary ``fields``.

    No-op (single bool check) when tracing is disabled. Each field is rendered
    on its own indented line so multi-line payloads (chunk text, tool I/O,
    prompts) stay readable.
    """
    if not _state.enabled:
        return
    lines = [f"[trace] {step}"]
    for key, value in fields.items():
        lines.append(f"    {key} = {_fmt_value(value, max_len=max_len)}")
    record = "\n".join(lines)
    for sink in _state.sinks:
        sink(record)
