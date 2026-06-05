"""Pytest configuration for ChatQEC."""

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir():
    """Path to the tests/fixtures directory (created lazily in later milestones)."""
    return Path(__file__).parent / "fixtures"
