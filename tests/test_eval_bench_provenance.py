"""Provenance header: real values where available, 'unavailable' where not, never a guess."""

from pathlib import Path

from chatqec.eval.benchmarks import provenance as prov

REPO = Path(__file__).resolve().parent.parent


def test_collect_records_commits_hashes_and_settings(tmp_path):
    pins = REPO / "eval/benchmarks/pins.yaml"
    gold = REPO / "eval/benchmarks/decoder_bench/benchmark_results.csv"
    p = prov.collect(
        chatqec_repo=REPO, tools_repo=tmp_path, pins=pins, gold=gold,
        model="muse-glimmer", model_url=None, settings={"seed": 42},
    )  # fmt: skip
    assert len(p["chatqec"]["commit"]) == 40
    assert p["tools"]["commit"] == prov.UNAVAILABLE  # tmp_path is not a git repo
    assert p["gold"]["sha256"] == prov.sha256_file(gold) and len(p["pins_sha256"]) == 64
    assert p["model"] == "muse-glimmer" and p["served_model"] == prov.UNAVAILABLE
    assert p["settings"] == {"seed": 42}


def test_served_model_is_unavailable_when_nothing_listens():
    assert prov.served_model("http://127.0.0.1:9/v1") == prov.UNAVAILABLE
