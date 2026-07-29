"""Chat page — streaming Q&A with cited evidence and tool output rendering."""

import os
from datetime import datetime
from pathlib import Path

import streamlit as st

from chatqec.trace import disable_trace, enable_trace
from chatqec_app.components import citation_card, tool_output
from chatqec_app.utils import (
    client_ip,
    get_captcha,
    get_chatqec,
    get_cost_budget,
    get_rate_limiter,
    utc_now,
)

st.title("Chat")

# --- verbose trace toggle ---
verbose = st.sidebar.toggle(
    "🔍 Verbose trace",
    value=False,
    help="Write every pipeline step (classify, rewrite, retrieve, synthesize, "
    "model/provider, MCP tool calls) with full payloads to a timestamped file "
    "under logs/ — same as the CLI --log-file. Not shown inline.",
)

# --- session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "captcha_verified" not in st.session_state:
    st.session_state.captcha_verified = False

# --- captcha gate (first question only) ---
captcha = get_captcha()
if captcha.enabled and not st.session_state.captcha_verified:
    st.warning("Please complete the captcha to ask your first question.")
    token = st.text_input(
        "Captcha token",
        help=(
            "Paste the hCaptcha response token (the embedded widget lives in "
            "production; in dev set HCAPTCHA_SECRET='' to disable)."
        ),
    )
    if token and captcha.verify(token):
        st.session_state.captcha_verified = True
        st.rerun()
    st.stop()


# --- render history ---
def _render_figures(paths: list[str]) -> None:
    """Render retrieved figure images whose files still exist on disk."""
    for path in paths:
        if Path(path).is_file():
            st.image(path, caption=path)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
        for c in msg.get("citations", []):
            citation_card.render(c)
        for t in msg.get("tool_calls", []):
            tool_output.render(t)
        _render_figures(msg.get("figures", []))

# --- accept new question (optionally with attached images) ---
submission = st.chat_input(
    "Ask a QEC question…",
    accept_file="multiple",
    file_type=["png", "jpg", "jpeg", "webp"],
)
question = submission.text if submission else None
uploaded = submission.files if submission else []
if submission and (question or uploaded):
    bot = get_chatqec()
    rl = get_rate_limiter()
    budget = get_cost_budget()
    ip = client_ip()
    now = utc_now()

    from chatqec.guards.rate_limit import LimitExceeded
    from chatqec.ingestion.images import ImageError, prepare_images

    try:
        rl.check_and_record(ip, action="ask", now=now)
    except LimitExceeded:
        st.error("Daily question limit (20/day) reached. Try again tomorrow.")
        st.stop()

    if budget.remaining(now=now) <= 0:
        st.error("Daily budget exhausted. Service degraded — try self-host instructions.")
        st.stop()

    images = []
    if uploaded:
        try:
            images = prepare_images([(f.getvalue(), f.type, f.name) for f in uploaded])
        except ImageError as e:
            st.error(str(e))
            st.stop()

    st.session_state.messages.append({"role": "user", "text": question or ""})
    with st.chat_message("user"):
        st.markdown(question or "")
        for im in images:
            st.image(im.data, width=240)

    # Write the execution trace to a logs/ file (same as the CLI), not the UI.
    # One timestamped file per Streamlit session; each question appends to it.
    trace_log_path: Path | None = None
    if verbose:
        if "trace_log_path" not in st.session_state:
            st.session_state["trace_log_path"] = (
                f"logs/chatqec-trace-{datetime.now():%Y%m%d-%H%M%S}.log"
            )
        trace_log_path = enable_trace(
            True, log_file=st.session_state["trace_log_path"], to_stderr=False
        )
    else:
        disable_trace()

    # When MCP tools are attached, ask_stream() can't stream token-by-token anyway
    # (run_tool_loop blocks until the whole tool loop finishes, then yields the
    # full text as one chunk — see Synthesizer.synthesize_stream). Calling it and
    # then bot.ask() again would re-run the entire pipeline including every tool
    # call twice, at random-sampled Stim circuits' expense: the two runs can
    # legitimately return different samples, so the text that streamed in and the
    # text that gets persisted/shown could silently disagree. Skip straight to the
    # single non-streaming call in that case. Same env var get_chatqec() uses to
    # decide whether to attach the MCP adapter in the first place.
    mcp_enabled = bool(os.environ.get("CHATQEC_MCP_SERVER"))
    with st.chat_message("assistant"):
        if mcp_enabled:
            with st.spinner("Running QEC tools…"):
                answer = bot.ask(question or "", images=images)
            st.markdown(answer.text)
        else:
            placeholder = st.empty()
            full = ""
            for chunk in bot.ask_stream(question or "", images=images):
                full += chunk.delta
                placeholder.markdown(full + "▌")
            placeholder.markdown(full)

            # Get final structured Answer from a non-stream call to harvest
            # citations/tools (the streaming path doesn't return them yet; M9
            # stays explicit about this limit).
            answer = bot.ask(question or "", images=images)

        # Rendered inside the same chat_message container as the answer text
        # (not after it closes) so a blocked-on-bot.ask() rerun never leaves a
        # prior turn's citation/tool widgets as an orphaned, stale-faded
        # leftover sitting below the next turn's spinner.
        for c in answer.citations:
            citation_card.render(c)
        for t in answer.tool_calls:
            tool_output.render(t)
        _render_figures(answer.figure_paths)
    if images:
        budget.charge(amount_usd=0.01 * len(images), now=now)

    if verbose:
        disable_trace()
        if trace_log_path is not None:
            st.caption(f"🔍 Execution trace written to `{trace_log_path}`")
    st.session_state.messages.append(
        {
            "role": "assistant",
            "text": answer.text,
            "citations": answer.citations,
            "tool_calls": answer.tool_calls,
            "figures": answer.figure_paths,
        }
    )

# --- footer: usage transparency ---
ip = client_ip()
rl = get_rate_limiter()
budget = get_cost_budget()
now = utc_now()
st.caption(
    f"asks remaining today: {rl.remaining(ip, action='ask', now=now)} · "
    f"budget remaining: ${budget.remaining(now=now):.2f}"
)
