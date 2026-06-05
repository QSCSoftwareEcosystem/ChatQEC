"""Pytest configuration for ChatQEC."""

import pytest


@pytest.fixture
def fixtures_dir():
    """Path to the tests/fixtures directory (created lazily in later milestones)."""
    from pathlib import Path
    return Path(__file__).parent / "fixtures"
