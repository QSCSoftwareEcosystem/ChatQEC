"""Ordered-fallback LLM client tests."""

from unittest.mock import MagicMock

import pytest

from chatqec.providers.fallback import Backend, FallbackLLMClient, _is_recoverable


def _ok_response(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    resp.stop_reason = "end_turn"
    return resp


def _client_ok(text: str = "ok"):
    c = MagicMock()
    c.messages.create.return_value = _ok_response(text)
    return c


def _client_raises(exc: Exception):
    c = MagicMock()
    c.messages.create.side_effect = exc
    return c


def test_primary_succeeds_secondary_not_called():
    primary = _client_ok("primary")
    secondary = _client_ok("secondary")
    fb = FallbackLLMClient(
        backends=[
            Backend(name="primary", client=primary),
            Backend(name="secondary", client=secondary),
        ],
        verbose=False,
    )
    resp = fb.messages.create(
        model="m", max_tokens=10, system="s", messages=[{"role": "user", "content": "q"}]
    )
    assert resp.content[0].text == "primary"
    primary.messages.create.assert_called_once()
    secondary.messages.create.assert_not_called()


def test_primary_credit_error_falls_to_secondary():
    primary = _client_raises(Exception("400 Your credit balance is too low"))
    secondary = _client_ok("backup")
    fb = FallbackLLMClient(
        backends=[
            Backend(name="primary", client=primary),
            Backend(name="secondary", client=secondary),
        ],
        verbose=False,
    )
    resp = fb.messages.create(
        model="m", max_tokens=10, system="s", messages=[{"role": "user", "content": "q"}]
    )
    assert resp.content[0].text == "backup"
    secondary.messages.create.assert_called_once()


def test_primary_403_falls_to_secondary():
    primary = _client_raises(Exception("403 Forbidden: missing permissions"))
    secondary = _client_ok("backup")
    fb = FallbackLLMClient(
        backends=[
            Backend(name="primary", client=primary),
            Backend(name="secondary", client=secondary),
        ],
        verbose=False,
    )
    resp = fb.messages.create(
        model="m", max_tokens=10, system=None, messages=[{"role": "user", "content": "q"}]
    )
    assert resp.content[0].text == "backup"


def test_non_recoverable_error_bubbles_up():
    primary = _client_raises(RuntimeError("connection reset by peer"))
    secondary = _client_ok("never reached")
    fb = FallbackLLMClient(
        backends=[
            Backend(name="primary", client=primary),
            Backend(name="secondary", client=secondary),
        ],
        verbose=False,
    )
    with pytest.raises(RuntimeError, match="connection reset"):
        fb.messages.create(
            model="m",
            max_tokens=10,
            system="s",
            messages=[{"role": "user", "content": "q"}],
        )
    secondary.messages.create.assert_not_called()


def test_all_backends_failing_raises_runtime_error():
    primary = _client_raises(Exception("403 Forbidden"))
    secondary = _client_raises(Exception("401 unauthorized"))
    fb = FallbackLLMClient(
        backends=[
            Backend(name="primary", client=primary),
            Backend(name="secondary", client=secondary),
        ],
        verbose=False,
    )
    with pytest.raises(RuntimeError, match="all LLM backends failed"):
        fb.messages.create(
            model="m",
            max_tokens=10,
            system="s",
            messages=[{"role": "user", "content": "q"}],
        )


def test_model_remap_translates_for_secondary():
    primary = _client_raises(Exception("403 credit balance too low"))
    secondary = _client_ok("done")
    fb = FallbackLLMClient(
        backends=[
            Backend(name="anthropic", client=primary),
            Backend(
                name="huggingface",
                client=secondary,
                model_remap={"claude-sonnet-4-6": "meta-llama/Llama-3.3-70B-Instruct"},
                default_model="meta-llama/Llama-3.3-70B-Instruct",
            ),
        ],
        verbose=False,
    )
    fb.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        system="s",
        messages=[{"role": "user", "content": "q"}],
    )
    secondary.messages.create.assert_called_once()
    kwargs = secondary.messages.create.call_args.kwargs
    assert kwargs["model"] == "meta-llama/Llama-3.3-70B-Instruct"


def test_default_model_used_when_remap_missing():
    primary = _client_raises(Exception("403"))
    secondary = _client_ok("done")
    fb = FallbackLLMClient(
        backends=[
            Backend(name="primary", client=primary),
            Backend(
                name="secondary",
                client=secondary,
                model_remap={},
                default_model="meta-llama/Llama-3.3-70B-Instruct",
            ),
        ],
        verbose=False,
    )
    fb.messages.create(
        model="some-unknown-claude-model",
        max_tokens=100,
        system="s",
        messages=[{"role": "user", "content": "q"}],
    )
    assert (
        secondary.messages.create.call_args.kwargs["model"] == "meta-llama/Llama-3.3-70B-Instruct"
    )


def test_tools_passed_through_to_chosen_backend():
    primary = _client_ok("with tools")
    fb = FallbackLLMClient(backends=[Backend(name="primary", client=primary)], verbose=False)
    fb.messages.create(
        model="m",
        max_tokens=100,
        system="s",
        messages=[{"role": "user", "content": "q"}],
        tools=[{"name": "stim_simulate", "input_schema": {}}],
    )
    assert "tools" in primary.messages.create.call_args.kwargs


def test_verbose_log_announces_backend(capsys):
    primary = _client_raises(Exception("403 Forbidden"))
    secondary = _client_ok("from secondary")
    fb = FallbackLLMClient(
        backends=[
            Backend(name="anthropic", client=primary),
            Backend(name="huggingface", client=secondary),
        ],
        verbose=True,
    )
    fb.messages.create(
        model="m", max_tokens=10, system=None, messages=[{"role": "user", "content": "q"}]
    )
    err = capsys.readouterr().err
    assert "anthropic" in err and "unavailable" in err
    assert "huggingface" in err


def test_empty_backends_rejected():
    with pytest.raises(ValueError):
        FallbackLLMClient(backends=[])


def test_is_recoverable_catches_expected_patterns():
    assert _is_recoverable(Exception("403 Forbidden"))
    assert _is_recoverable(Exception("Your credit balance is too low"))
    assert _is_recoverable(Exception("429 Too Many Requests"))
    assert _is_recoverable(Exception("401 Unauthorized"))
    assert _is_recoverable(Exception("insufficient quota"))
    assert not _is_recoverable(Exception("503 Service Unavailable"))
    assert not _is_recoverable(Exception("connection reset"))
    assert not _is_recoverable(Exception("malformed request body"))
