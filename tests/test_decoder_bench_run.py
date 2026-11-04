"""_decoder_bench_run: decoder-bench's own pipeline (vendored) on one memory config."""

import pytest

from chatqec_mcp_tools.tools import _decoder_bench_run as dbr


def test_circuit_is_the_vendored_benchmark_circuit():
    from chatqec_mcp_tools.tools._vendor.decoder_bench import generators as g

    c = dbr.build_circuit("surface", 3, 0.005)
    assert dbr.circuit_sha256(c) == dbr.circuit_sha256(g.gen_surface_circuit((3, 0.005, "z")))


@pytest.mark.parametrize("decoder", ["pymatching", "belieffind", "bplsd", "mwpf"])
def test_decoders_run_on_shared_syndromes(decoder):
    c = dbr.build_circuit("surface", 3, 0.005)
    (r,) = dbr.run(c, [decoder], shots=300, seed=42)
    assert r.ok, r.note
    assert r.shots == 300 and 0 <= r.num_errors <= 300


def test_same_seed_same_errors():
    c = dbr.build_circuit("surface", 3, 0.005)
    a = dbr.run(c, ["pymatching"], shots=1000, seed=5)[0]
    b = dbr.run(c, ["pymatching"], shots=1000, seed=5)[0]
    assert a.num_errors == b.num_errors


def test_unsupported_distance_is_refused():
    with pytest.raises(ValueError, match="6, 10, 12"):
        dbr.build_circuit("qldpc", 7, 0.001)


def test_missing_decoder_package_is_reported_not_raised(monkeypatch):
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "relay_bp":
            raise ImportError(name=name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    (r,) = dbr.run(dbr.build_circuit("surface", 3, 0.005), ["relay-bp"], shots=10, seed=1)
    assert not r.ok and "relay_bp" in r.note
