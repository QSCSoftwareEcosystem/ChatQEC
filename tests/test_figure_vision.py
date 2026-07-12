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


def test_available_returns_bool_without_loading_model(monkeypatch):
    # Cheap probe: returns a plain bool and never touches the weights.
    assert isinstance(LlamaVisionDescriber.available(), bool)


def test_available_false_when_torch_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    assert LlamaVisionDescriber.available() is False
