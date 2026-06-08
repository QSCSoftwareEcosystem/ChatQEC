"""Content-hash utilities used by registry refresh logic to detect updates."""

from pathlib import Path

from chatqec.ingestion.hashing import sha256_bytes, sha256_text, sha256_file


def test_sha256_text_is_deterministic():
    h1 = sha256_text("hello, qec")
    h2 = sha256_text("hello, qec")
    assert h1 == h2
    assert len(h1) == 64  # hex digest length


def test_sha256_text_known_vector():
    # echo -n "" | sha256sum  →  e3b0c44... (the well-known empty-string digest)
    assert sha256_text("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_sha256_bytes_normalizes_to_text_for_ascii():
    assert sha256_text("abc") == sha256_bytes(b"abc")


def test_sha256_file_matches_text_for_small_file(tmp_path: Path):
    p = tmp_path / "fixture.txt"
    p.write_text("ChatQEC", encoding="utf-8")
    assert sha256_file(p) == sha256_text("ChatQEC")


def test_sha256_file_streams_large_input(tmp_path: Path):
    # Build a 5 MB file from a repeated pattern; verify file hash equals one-shot hash.
    chunk = b"qec-" * (1024 * 256)  # 1 MiB-ish
    big = chunk * 5
    p = tmp_path / "big.bin"
    p.write_bytes(big)
    assert sha256_file(p) == sha256_bytes(big)
