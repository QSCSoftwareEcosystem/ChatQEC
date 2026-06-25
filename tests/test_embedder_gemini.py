"""Tests for GeminiEmbedder API-key configuration."""

import pytest
from unittest.mock import MagicMock

from chatqec.embedding.base import EmbedderConfig
from chatqec.embedding.gemini import GeminiEmbedder


def _cfg() -> EmbedderConfig:
    return EmbedderConfig(
        provider="gemini",
        model="gemini-embedding-001",
        batch_size=8,
        local_model="",
    )


def test_configures_client_from_gemini_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key-123")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    mock_genai = MagicMock()
    monkeypatch.setattr("chatqec.embedding.gemini.genai", mock_genai)

    GeminiEmbedder(_cfg())

    mock_genai.configure.assert_called_once_with(api_key="gem-key-123")


def test_falls_back_to_google_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "goog-key-456")
    mock_genai = MagicMock()
    monkeypatch.setattr("chatqec.embedding.gemini.genai", mock_genai)

    GeminiEmbedder(_cfg())

    mock_genai.configure.assert_called_once_with(api_key="goog-key-456")


def test_raises_clear_error_without_any_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr("chatqec.embedding.gemini.genai", MagicMock())

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiEmbedder(_cfg())
