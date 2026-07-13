"""The system prompt should allow cited web-source answers when the corpus is weak."""

from chatqec.conversation.prompts import SYNTHESIZER_SYSTEM_PROMPT


def test_prompt_allows_web_fallback_with_disclosure():
    p = SYNTHESIZER_SYSTEM_PROMPT.lower()
    assert "web" in p
    # must instruct to disclose web reliance and still refuse when nothing is found
    assert "refuse" in p
