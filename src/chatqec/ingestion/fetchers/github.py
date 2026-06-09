"""GitHub fetcher — operates on a dict of {path: text} after a LlamaIndex pull.

The actual repo download uses LlamaIndex's `GithubRepositoryReader`; this
module's parser is pure so it can be unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass

from chatqec.ingestion.chunk import Chunk
from chatqec.ingestion.chunkers.math_aware import chunk_text
from chatqec.ingestion.ids import github_id


@dataclass(frozen=True)
class GitHubMetadata:
    org: str
    repo: str
    commit_sha: str
    license: str


def files_to_chunks(
    md: GitHubMetadata,
    *,
    files: dict[str, str],
    target_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    source_id = github_id(md.org, md.repo, md.commit_sha)
    out: list[Chunk] = []
    idx = 0
    for path, text in files.items():
        if not text.strip():
            continue
        for piece in chunk_text(text, target_tokens=target_tokens, overlap_tokens=overlap_tokens):
            out.append(
                Chunk(
                    chunk_id=f"{source_id}#{idx}",
                    source_id=source_id,
                    chunk_index=idx,
                    text=piece,
                    title=f"{md.org}/{md.repo}",
                    source_type="github",
                    authority_tier="blog",
                    anchor=f"path={path}",
                    metadata={"path": path, "license": md.license},
                )
            )
            idx += 1
    return out
