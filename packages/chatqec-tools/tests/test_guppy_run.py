"""guppy_run: run a Guppy program on Selene. Off by default; source-gated; isolated child."""

import importlib.metadata
import os

import pytest
from pydantic import ValidationError

from chatqec_mcp_tools.tools import guppy_run as gr
from chatqec_mcp_tools.tools.guppy_run import GuppyRunInput, guppy_run


def _guppy_major() -> int:
    try:
        return int(importlib.metadata.version("guppylang").split(".")[0])
    except importlib.metadata.PackageNotFoundError:
        return 0


# guppylang >= 1.0 returns a Measurement future that result() cannot take; read it first.
_READ = ".read()" if _guppy_major() >= 1 else ""
HEADER = (
    "from guppylang import guppy\n"
    "from guppylang.std.builtins import result\n"
    "from guppylang.std.quantum import qubit, h, cx, measure\n\n"
)
BELL = (
    HEADER
    + "@guppy\ndef main() -> None:\n"
    + "    q0 = qubit()\n    q1 = qubit()\n    h(q0)\n    cx(q0, q1)\n"
    + f'    result("c0", measure(q0){_READ})\n    result("c1", measure(q1){_READ})\n'
)


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("CHATQEC_ALLOW_GUPPY_EXEC", "1")


def test_disabled_by_default_and_says_how_to_enable(monkeypatch):
    monkeypatch.delenv("CHATQEC_ALLOW_GUPPY_EXEC", raising=False)
    out = guppy_run(GuppyRunInput(source=BELL))
    assert out.ok is False
    assert "CHATQEC_ALLOW_GUPPY_EXEC" in out.error_message and "disabled" in out.error_message


def test_flag_must_be_exactly_one(monkeypatch):
    monkeypatch.setenv("CHATQEC_ALLOW_GUPPY_EXEC", "true")
    assert guppy_run(GuppyRunInput(source=BELL)).ok is False


def test_hostile_source_is_rejected_and_never_executed(enabled, tmp_path):
    canary = tmp_path / "pwned"
    src = f"import os\nos.system('touch {canary}')\n" + BELL
    out = guppy_run(GuppyRunInput(source=src))
    assert out.ok is False and "import" in out.error_message.lower()
    assert not canary.exists()


def test_source_length_is_capped():
    with pytest.raises(ValidationError):
        GuppyRunInput(source="x" * 30_000)


def test_missing_extra_points_at_the_guppy_extra(enabled, monkeypatch):
    monkeypatch.setattr(gr.importlib.util, "find_spec", lambda name: None)
    out = guppy_run(GuppyRunInput(source=BELL))
    assert out.ok is False and "[guppy]" in out.error_message


pytest.importorskip("guppylang")
pytest.importorskip("selene_sim")


def test_bell_program_gives_only_correlated_outcomes_and_is_seed_reproducible(enabled):
    a = guppy_run(GuppyRunInput(source=BELL, shots=100, n_qubits=2, seed=7))
    b = guppy_run(GuppyRunInput(source=BELL, shots=100, n_qubits=2, seed=7))
    assert a.ok, a.error_message
    assert a.registers == ["c0", "c1"] and a.shots == 100
    assert sum(a.counts.values()) == 100
    assert set(a.counts) <= {"0,0", "1,1"} and len(a.counts) == 2
    assert a.counts == b.counts


def test_a_runtime_failure_is_reported_with_the_stderr_tail(enabled):
    too_many = (
        HEADER
        + "@guppy\ndef main() -> None:\n    a = qubit()\n    b = qubit()\n    c = qubit()\n    result('x', measure(a))\n"
    )
    out = guppy_run(GuppyRunInput(source=too_many, n_qubits=1, shots=1))
    assert out.ok is False and out.error_message


def test_a_hung_program_is_stopped_at_the_time_limit(enabled, monkeypatch):
    monkeypatch.setattr(gr, "_WALL_SECONDS", 25)
    forever = (
        HEADER + "@guppy\ndef main() -> None:\n    q = qubit()\n    while True:\n        h(q)\n"
    )
    out = guppy_run(GuppyRunInput(source=forever, n_qubits=1, shots=1))
    assert out.ok is False and "time limit" in out.error_message
    assert os.environ.get("CHATQEC_ALLOW_GUPPY_EXEC") == "1"
