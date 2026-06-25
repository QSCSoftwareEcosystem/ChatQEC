"""The CLI loads a local .env so GEMINI_API_KEY etc. are honored."""


def test_main_loads_dotenv(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "chatqec.cli.load_dotenv", lambda *a, **k: calls.append(True), raising=False
    )

    from chatqec.cli import main

    main.callback()

    assert calls == [True]
