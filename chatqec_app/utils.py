"""Shared helpers for Streamlit pages."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from chatqec.api import ChatQEC
from chatqec.cli import _build_chatqec
from chatqec.guards.captcha import HCaptchaVerifier
from chatqec.guards.cost_budget import CostBudget
from chatqec.guards.rate_limit import RateLimiter


def _build_bot() -> ChatQEC:
    # Streamlit doesn't go through the CLI's main(), so load .env here to pick up
    # provider keys (GEMINI_API_KEY, ANTHROPIC_API_KEY, ...).
    load_dotenv()
    # Reuse the CLI builder so the app shares `chatqec ask` wiring. Enable Stim
    # tools by setting CHATQEC_MCP_SERVER (e.g. "chatqec-mcp-tools"); off by
    # default so deployments without the MCP server keep working.
    mcp_server = os.environ.get("CHATQEC_MCP_SERVER") or None
    return _build_chatqec(mcp_server=mcp_server)


@st.cache_resource
def get_chatqec() -> ChatQEC:
    return _build_bot()


@st.cache_resource
def get_rate_limiter() -> RateLimiter:
    db = Path(os.environ.get("CHATQEC_GUARD_DB", "data/chatqec.db"))
    return RateLimiter(
        db_path=db,
        limits={
            "ask": (20, "day"),  # spec §12.1
            "tool": (5, "hour"),
        },
    )


@st.cache_resource
def get_cost_budget() -> CostBudget:
    db = Path(os.environ.get("CHATQEC_GUARD_DB", "data/chatqec.db"))
    cap = float(os.environ.get("CHATQEC_DAILY_BUDGET_USD", "5.00"))
    return CostBudget(db_path=db, daily_usd_cap=cap)


@st.cache_resource
def get_captcha() -> HCaptchaVerifier:
    return HCaptchaVerifier(secret=os.environ.get("HCAPTCHA_SECRET", ""))


def client_ip() -> str:
    # Streamlit doesn't expose remote IP directly; use session id as a coarse
    # per-user key when no IP header is present.
    ctx = st.runtime.scriptrunner.get_script_run_ctx()  # type: ignore[attr-defined]
    return getattr(ctx, "session_id", "local-dev")


def utc_now() -> datetime:
    return datetime.now(tz=UTC)
