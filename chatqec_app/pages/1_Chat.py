"""Chat page — streaming Q&A with cited evidence and tool output rendering."""

import streamlit as st

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
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
        for c in msg.get("citations", []):
            citation_card.render(c)
        for t in msg.get("tool_calls", []):
            tool_output.render(t)

# --- accept new question ---
question = st.chat_input("Ask a QEC question…")
if question:
    bot = get_chatqec()
    rl = get_rate_limiter()
    budget = get_cost_budget()
    ip = client_ip()
    now = utc_now()

    from chatqec.guards.rate_limit import LimitExceeded

    try:
        rl.check_and_record(ip, action="ask", now=now)
    except LimitExceeded:
        st.error("Daily question limit (20/day) reached. Try again tomorrow.")
        st.stop()

    if budget.remaining(now=now) <= 0:
        st.error("Daily budget exhausted. Service degraded — try self-host instructions.")
        st.stop()

    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full = ""
        for chunk in bot.ask_stream(question):
            full += chunk.delta
            placeholder.markdown(full + "▌")
        placeholder.markdown(full)

    # Get final structured Answer from a non-stream call to harvest citations/tools
    # (the streaming path doesn't return them yet; M9 stays explicit about this limit)
    answer = bot.ask(question)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "text": answer.text,
            "citations": answer.citations,
            "tool_calls": answer.tool_calls,
        }
    )
    for c in answer.citations:
        citation_card.render(c)
    for t in answer.tool_calls:
        tool_output.render(t)

# --- footer: usage transparency ---
ip = client_ip()
rl = get_rate_limiter()
budget = get_cost_budget()
now = utc_now()
st.caption(
    f"asks remaining today: {rl.remaining(ip, action='ask', now=now)} · "
    f"budget remaining: ${budget.remaining(now=now):.2f}"
)
