"""The system prompt should advertise the QEC computation tools by name."""

from chatqec.conversation.prompts import SYNTHESIZER_SYSTEM_PROMPT


def test_prompt_mentions_all_qec_tools():
    for name in (
        "stim_simulate",
        "pymatching_decode",
        "threshold_sweep",
        "code_params",
        "decoder_compare",
        "ibm_add_pauli_checks",
        "ibm_ft_compile",
        "deltakit_memory",
        "guppy_run",
        "ft_primitive_simulate",
    ):
        assert name in SYNTHESIZER_SYSTEM_PROMPT


def test_prompt_tells_the_model_to_retry_a_guppy_compile_error_once_and_say_what_changed():
    # Regression guard: guppylang 1.x rejects result("c", measure(q)) and the error names the
    # fix (.read()); the model reported the error but never retried the corrected program.
    assert ".read()" in SYNTHESIZER_SYSTEM_PROMPT
    assert "once more" in SYNTHESIZER_SYSTEM_PROMPT


def test_refusal_rule_does_not_override_tools():
    """Gemma 4 answered tool questions with the refusal line ("I don't have evidence
    in the knowledge base") or "I do not have a tool", because the refusal rule
    came first and had no exception for questions a listed tool can answer."""
    p = " ".join(SYNTHESIZER_SYSTEM_PROMPT.split())  # ignore line wrapping
    refusal_at = p.index("I don't have evidence in the knowledge base")
    exception = "unless one of the available tools can answer it"
    assert exception in p[refusal_at - 300 : refusal_at + 300]
    assert "Never say you lack a tool that is listed" in p
