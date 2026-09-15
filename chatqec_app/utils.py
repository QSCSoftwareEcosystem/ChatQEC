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
from chatqec_app.qdrant_boot import ensure_qdrant


def render_branding() -> None:
    """Show the ChatQEC logo at the top of the sidebar, above the page nav.
    Call once per page.

    st.logo() docks to that top-of-sidebar slot, but caps the image at
    20/24/32px tall (size="small"/"medium"/"large") regardless of the
    source image's resolution -- too small against the full sidebar width.
    The CSS override below lifts that cap so the logo fills the sidebar's
    width instead, while keeping st.logo's placement above the nav.
    """
    st.logo("docs/images/chatqec_logo_final.svg", size="large")
    st.html(
        """
        <style>
        [data-testid="stSidebarHeader"] {
            height: auto;
            align-items: center;
        }
        [data-testid="stSidebarHeader"] [data-testid="stLogoLink"] {
            width: 100%;
        }
        [data-testid="stSidebarLogo"] {
            height: auto !important;
            max-height: none !important;
            width: 100% !important;
            max-width: 100% !important;
        }
        </style>
        """
    )


def _build_bot() -> ChatQEC:
    # Streamlit doesn't go through the CLI's main(), so load .env here to pick up
    # provider keys (GEMINI_API_KEY, ANTHROPIC_API_KEY, ...).
    load_dotenv()
    # Reuse the CLI builder so the app shares `chatqec ask` wiring. Enable Stim
    # tools by setting CHATQEC_MCP_SERVER (e.g. "chatqec-mcp-tools"); off by
    # default so deployments without the MCP server keep working.
    mcp_server = os.environ.get("CHATQEC_MCP_SERVER") or None
    # The retriever connects to Qdrant during construction, so a missing store
    # would crash the page on load. Start a local one first when needed.
    ensure_qdrant()
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
