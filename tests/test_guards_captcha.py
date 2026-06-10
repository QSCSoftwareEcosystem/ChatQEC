"""hCaptcha verifier tests (HTTP call mocked)."""

from unittest.mock import MagicMock

from chatqec.guards.captcha import HCaptchaVerifier


def test_captcha_verifier_returns_true_on_success(monkeypatch):
    fake_post = MagicMock()
    fake_post.return_value.json.return_value = {"success": True, "challenge_ts": "..."}
    fake_post.return_value.raise_for_status.return_value = None
    monkeypatch.setattr("chatqec.guards.captcha.requests.post", fake_post)

    v = HCaptchaVerifier(secret="SECRET")
    assert v.verify("user-token") is True
    fake_post.assert_called_once()


def test_captcha_verifier_returns_false_on_failure(monkeypatch):
    fake_post = MagicMock()
    fake_post.return_value.json.return_value = {
        "success": False,
        "error-codes": ["bad-token"],
    }
    fake_post.return_value.raise_for_status.return_value = None
    monkeypatch.setattr("chatqec.guards.captcha.requests.post", fake_post)

    v = HCaptchaVerifier(secret="SECRET")
    assert v.verify("bad-token") is False


def test_captcha_verifier_disabled_when_secret_empty():
    """No secret configured — captcha is disabled and always passes."""
    v = HCaptchaVerifier(secret="")
    assert v.enabled is False
    assert v.verify("anything") is True
