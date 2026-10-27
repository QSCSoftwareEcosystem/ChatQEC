"""The system prompt should advertise the QEC computation tools by name."""

from chatqec.conversation.prompts import SYNTHESIZER_SYSTEM_PROMPT


def test_prompt_mentions_all_qec_tools():
    for name in (
        "stim_simulate",
        "pymatching_decode",
        "threshold_sweep",
        "code_params",
        "decoder_compare",
    ):
        assert name in SYNTHESIZER_SYSTEM_PROMPT
