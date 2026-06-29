"""Ordered-fallback LLM client.

Tries each backend in order; on *recoverable* failures (insufficient credit,
auth, rate-limit, 403/429) falls through to the next. Other errors (5xx,
network, malformed args) bubble up — those are real outages, not budget issues.

Every call logs which backend served it to stderr so users always know which
provider produced an answer. There is no silent quality drop.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from chatqec.providers.base import MessageResponse
from chatqec.trace import is_enabled as trace_is_enabled
from chatqec.trace import trace


@dataclass(frozen=True)
class Backend:
    """One backend the fallback client can try."""

    name: str
    client: Any
    # Maps a logical "primary" model name to this backend's equivalent.
    # E.g. Anthropic role "claude-sonnet-4-6" -> HF "meta-llama/Llama-3.3-70B-Instruct".
    model_remap: Mapping[str, str] = field(default_factory=dict)
    # Used when ``model_remap`` doesn't have an entry for the requested model.
    default_model: str | None = None


def _is_recoverable(exc: BaseException) -> bool:
    """Catch errors that mean 'this backend is unavailable, try the next'."""
    msg = str(exc).lower()
    markers = (
        "credit balance",
        "credit_balance",
        "insufficient",
        "billing",
        "quota",
        "403",
        "forbidden",
        "401",
        "unauthor",
        "rate limit",
        "rate_limit",
        "429",
        "too many requests",
        "permission",
    )
    return any(m in msg for m in markers)


class _FallbackMessages:
    def __init__(self, owner: FallbackLLMClient) -> None:
        self._owner = owner

    def _log(self, line: str) -> None:
        # When step tracing is on, the same provider events are captured as
        # trace records (to the log file), so skip the stderr line to keep the
        # terminal clean and avoid duplicates.
        if self._owner.verbose and not trace_is_enabled():
            sys.stderr.write(f"[chatqec] {line}\n")
            sys.stderr.flush()

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> MessageResponse:
        last_err: BaseException | None = None
        attempted: list[str] = []
        for be in self._owner.backends:
            if be.name in self._owner.tripped:
                continue  # circuit breaker: skip backends already known-unavailable
            attempted.append(be.name)
            eff_model = be.model_remap.get(model) or be.default_model or model
            kwargs: dict[str, Any] = {
                "model": eff_model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": messages or [],
            }
            if tools is not None:
                kwargs["tools"] = tools
            try:
                resp: MessageResponse = be.client.messages.create(**kwargs)
            except Exception as e:
                if not _is_recoverable(e):
                    raise
                last_err = e
                self._owner.tripped.add(be.name)  # don't retry this backend again
                self._log(
                    f"backend={be.name} unavailable ({type(e).__name__}: "
                    f"{str(e)[:120].strip()}) → trying next (skipping it from now on)"
                )
                trace(
                    "provider.backend_unavailable",
                    backend=be.name,
                    error=f"{type(e).__name__}: {str(e)[:120].strip()}",
                )
                continue
            self._log(f"backend={be.name} model={eff_model}")
            trace(
                "provider.backend_selected", backend=be.name, requested_model=model, model=eff_model
            )
            return resp
        raise RuntimeError(
            f"all LLM backends failed (tried: {', '.join(attempted)}); " f"last error: {last_err}"
        ) from last_err


class FallbackLLMClient:
    """Anthropic-shaped facade that delegates to an ordered list of backends."""

    def __init__(self, *, backends: list[Backend], verbose: bool = True) -> None:
        if not backends:
            raise ValueError("FallbackLLMClient requires at least one backend")
        self.backends = backends
        self.verbose = verbose
        # Names of backends that hit a recoverable error and are skipped from
        # then on (circuit breaker) — avoids re-trying/re-logging every call.
        self.tripped: set[str] = set()
        self.messages = _FallbackMessages(self)
