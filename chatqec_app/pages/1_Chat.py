"""Chat page — streaming Q&A with cited evidence and tool output rendering."""

import os
from datetime import datetime
from pathlib import Path

import streamlit as st

from chatqec.conversation.memory import ConversationMemory
from chatqec.trace import enable_trace
from chatqec_app.components import citation_card, tool_output
from chatqec_app.utils import (
    client_ip,
    get_captcha,
    get_chatqec,
    get_cost_budget,
    get_rate_limiter,
    render_branding,
    utc_now,
)

render_branding()
st.caption(
    "Ask cited QEC questions — grounded in curated papers, textbooks, and "
    "vendor docs, with real circuit simulation and IBM hardware tools when enabled."
)

# Answers use markdown headings (##, ###) to structure long tool-backed
# breakdowns (backend tables, calibration sections, ...). Streamlit's default
# heading sizes are tuned for a full-page document, not a chat bubble, so they
# read as oversized here -- scale them down without touching the rest of the
# page's headings (e.g. the h1 "Chat" title used to be here, now gone, but
# other pages still use normal-sized headings).
st.html(
    """
    <style>
    [data-testid="stChatMessageContent"] h1 { font-size: 1.3rem; }
    [data-testid="stChatMessageContent"] h2 { font-size: 1.15rem; }
    [data-testid="stChatMessageContent"] h3 { font-size: 1.05rem; }
    [data-testid="stChatMessageContent"] h4,
    [data-testid="stChatMessageContent"] h5,
    [data-testid="stChatMessageContent"] h6 { font-size: 1rem; }
    </style>
    """
)

EXAMPLE_QUESTIONS = [
    "What is the surface code and how does it detect errors?",
    "How does the threshold theorem relate to the break-even point demonstrated on real hardware?",
    "What IBM Quantum backends do I have access to right now?",
    "What is the stabilizer formalism and why is it central to quantum error correction?",
]

# --- session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
# The ChatQEC instance is cached and shared by every visitor, so conversation
# history must live in this session, never on the shared instance.
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = ConversationMemory()
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
        citation_card.render_all(msg.get("citations", []))
        for t in msg.get("tool_calls", []):
            tool_output.render(t)
        _render_figures(msg.get("figures", []))

# --- example question chips (first turn only) ---
if not st.session_state.messages:
    st.caption("Try asking:")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, example in zip(cols, EXAMPLE_QUESTIONS, strict=True):
        if col.button(example, use_container_width=True):
            st.session_state["queued_question"] = example
            st.rerun()

# --- accept new question (optionally with attached images) ---
submission = st.chat_input(
    "Ask a QEC question…",
    accept_file="multiple",
    file_type=["png", "jpg", "jpeg", "webp"],
)
queued_question = st.session_state.pop("queued_question", None)
question = (submission.text if submission else None) or queued_question
uploaded = submission.files if submission else []
if (submission or queued_question) and (question or uploaded):
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

    # Always write the execution trace to a logs/ file for debugging (same as the
    # CLI's --log-file) — never shown in the UI. One timestamped file per
    # Streamlit session; each question appends to it.
    if "trace_log_path" not in st.session_state:
        st.session_state["trace_log_path"] = (
            f"logs/chatqec-trace-{datetime.now():%Y%m%d-%H%M%S}.log"
        )
    enable_trace(True, log_file=st.session_state["trace_log_path"], to_stderr=False)

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
                answer = bot.ask(
                    question or "", images=images, memory=st.session_state.conversation_memory
                )
            st.markdown(answer.text)
        else:
            placeholder = st.empty()
            full = ""
            for chunk in bot.ask_stream(
                question or "", images=images, memory=st.session_state.conversation_memory
            ):
                full += chunk.delta
                placeholder.markdown(full + "▌")
            placeholder.markdown(full)

            # Get final structured Answer from a non-stream call to harvest
            # citations/tools (the streaming path doesn't return them yet; M9
            # stays explicit about this limit).
            answer = bot.ask(
                question or "", images=images, memory=st.session_state.conversation_memory
            )

        # Rendered inside the same chat_message container as the answer text
        # (not after it closes) so a blocked-on-bot.ask() rerun never leaves a
        # prior turn's citation/tool widgets as an orphaned, stale-faded
        # leftover sitting below the next turn's spinner.
        citation_card.render_all(answer.citations)
        for t in answer.tool_calls:
            tool_output.render(t)
        _render_figures(answer.figure_paths)
    if images:
        budget.charge(amount_usd=0.01 * len(images), now=now)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "text": answer.text,
            "citations": answer.citations,
            "tool_calls": answer.tool_calls,
            "figures": answer.figure_paths,
        }
    )
