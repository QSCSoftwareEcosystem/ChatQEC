"""Seeded tool-result cache: identical seeded calls are computed once across benchmark runs."""

from chatqec_mcp_tools.result_cache import cached_call, source_hash


def _counting():
    calls = []

    def handler(args):
        calls.append(dict(args))
        return {"ok": True, "value": args.get("x", 0) * 2}

    return handler, calls


def test_seeded_call_is_computed_once(tmp_path):
    handler, calls = _counting()
    a = cached_call("t", {"x": 3, "seed": 42}, handler, cache_dir=tmp_path, code_hash="h")
    b = cached_call("t", {"seed": 42, "x": 3}, handler, cache_dir=tmp_path, code_hash="h")
    assert len(calls) == 1 and a["value"] == b["value"] == 6
    assert "_cache_hit" not in a and b["_cache_hit"] is True


def test_unseeded_calls_are_never_cached(tmp_path):
    handler, calls = _counting()
    for args in ({"x": 1}, {"x": 1, "seed": None}):
        cached_call("t", args, handler, cache_dir=tmp_path, code_hash="h")
        cached_call("t", args, handler, cache_dir=tmp_path, code_hash="h")
    assert len(calls) == 4


def test_a_code_change_invalidates_the_cache(tmp_path):
    handler, calls = _counting()
    cached_call("t", {"x": 1, "seed": 1}, handler, cache_dir=tmp_path, code_hash="v1")
    cached_call("t", {"x": 1, "seed": 1}, handler, cache_dir=tmp_path, code_hash="v2")
    assert len(calls) == 2


def test_failed_results_are_not_cached(tmp_path):
    calls = []

    def flaky(args):
        calls.append(1)
        return {"ok": False, "error_message": "transient"}

    cached_call("t", {"seed": 1}, flaky, cache_dir=tmp_path, code_hash="h")
    cached_call("t", {"seed": 1}, flaky, cache_dir=tmp_path, code_hash="h")
    assert len(calls) == 2


def test_no_cache_dir_means_no_caching():
    handler, calls = _counting()
    cached_call("t", {"seed": 1}, handler, cache_dir=None, code_hash="h")
    cached_call("t", {"seed": 1}, handler, cache_dir=None, code_hash="h")
    assert len(calls) == 2


def test_source_hash_is_stable_and_changes_with_the_code(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "a.py").write_text("x = 1\n")
    h1 = source_hash(pkg)
    assert h1 == source_hash(pkg)
    (pkg / "a.py").write_text("x = 2\n")
    assert source_hash(pkg) != h1
