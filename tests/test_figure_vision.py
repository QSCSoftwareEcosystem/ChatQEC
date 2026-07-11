from chatqec.ingestion.figure_vision import FIGURE_PROMPT, LlamaVisionDescriber


def test_prompt_targets_qec_figures():
    p = FIGURE_PROMPT.lower()
    assert "figure" in p
    assert "equation" in p or "label" in p  # asks for OCR-worthy detail


def test_describer_is_a_figuredescriber():
    # Constructing must not load the model (lazy); it only stores config.
    d = LlamaVisionDescriber(device="cpu")
    assert hasattr(d, "describe")
    assert d.model_id == "meta-llama/Llama-3.2-11B-Vision-Instruct"
