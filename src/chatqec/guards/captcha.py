"""hCaptcha verifier (free tier).

Disabled (always passes) when secret is empty — convenient for local dev.
"""

from __future__ import annotations

import requests


class HCaptchaVerifier:
    VERIFY_URL = "https://hcaptcha.com/siteverify"

    def __init__(self, *, secret: str, timeout_seconds: int = 10) -> None:
        self.secret = secret
        self.timeout_seconds = timeout_seconds

    @property
    def enabled(self) -> bool:
        return bool(self.secret)

    def verify(self, token: str) -> bool:
        if not self.enabled:
            return True
        try:
            resp = requests.post(
                self.VERIFY_URL,
                data={"secret": self.secret, "response": token},
                timeout=self.timeout_seconds,
            )
            resp.raise_for_status()
            return bool(resp.json().get("success", False))
        except (requests.RequestException, ValueError):
            return False
