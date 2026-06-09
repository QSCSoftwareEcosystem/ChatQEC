"""Watchlist YAML schema (sources/watchlist.yaml)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class ArxivWatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    categories: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    authors: list[str] = Field(default_factory=list)


class GitHubWatchEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo: str

    @field_validator("repo")
    @classmethod
    def _two_segments(cls, v: str) -> str:
        if v.count("/") != 1 or v.startswith("/") or v.endswith("/"):
            raise ValueError(f"github repo must be 'org/name': {v!r}")
        return v


class YouTubeWatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    channels: list[str] = Field(default_factory=list)


class Watchlist(BaseModel):
    model_config = ConfigDict(extra="forbid")
    arxiv: ArxivWatch = Field(default_factory=ArxivWatch)
    github: list[GitHubWatchEntry] = Field(default_factory=list)
    youtube: YouTubeWatch = Field(default_factory=YouTubeWatch)
    rss: list[HttpUrl] = Field(default_factory=list)


def load_watchlist(path: Path) -> Watchlist:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Watchlist(**data)
