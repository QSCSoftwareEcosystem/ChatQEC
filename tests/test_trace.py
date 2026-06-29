"""Tracing facade: switchable, sink-capturing, full-payload step records."""

from chatqec import trace as trace_mod
from chatqec.trace import (
    current_log_path,
    disable_trace,
    enable_trace,
    is_enabled,
    trace,
)


def teardown_function() -> None:
    # Tracing state is process-global; reset after every test so ordering
    # can't leak an enabled tracer into an unrelated test.
    disable_trace()


def test_disabled_by_default_is_noop():
    captured: list[str] = []
    enable_trace(False, sink=captured.append, to_stderr=False)
    trace("step.one", key="value")
    assert captured == []
    assert is_enabled() is False


def test_enabled_emits_to_sink():
    captured: list[str] = []
    enable_trace(sink=captured.append, to_stderr=False)
    trace("classify.verdict", on_topic=True, refusal="")
    assert is_enabled() is True
    assert len(captured) == 1
    record = captured[0]
    assert record.startswith("[trace] classify.verdict")
    assert "on_topic = True" in record
    assert "refusal = " in record


def test_disable_trace_stops_emission():
    captured: list[str] = []
    enable_trace(sink=captured.append, to_stderr=False)
    trace("first")
    disable_trace()
    trace("second")
    assert [r.splitlines()[0] for r in captured] == ["[trace] first"]


def test_full_payload_not_truncated_by_default():
    captured: list[str] = []
    enable_trace(sink=captured.append, to_stderr=False)
    big = "x" * 5000
    trace("retrieval.chunk", text=big)
    assert big in captured[0]


def test_max_len_truncates_when_requested():
    captured: list[str] = []
    enable_trace(sink=captured.append, to_stderr=False)
    trace("retrieval.chunk", max_len=10, text="y" * 100)
    assert "yyyyyyyyyy… (+90 more chars)" in captured[0]


def test_multiline_payload_is_indented():
    captured: list[str] = []
    enable_trace(sink=captured.append, to_stderr=False)
    trace("synthesizer.context", block="line1\nline2")
    # The continuation line is indented under the key, not flush-left.
    assert "\n        line1\n        line2" in captured[0]


def test_module_level_state_visible_through_import():
    enable_trace(to_stderr=False)
    assert trace_mod.is_enabled() is True


def test_log_file_writes_readable_records(tmp_path):
    log = tmp_path / "nested" / "trace.log"
    returned = enable_trace(log_file=log)
    # Parent dirs are created and the resolved path is returned.
    assert returned == log.resolve()
    assert current_log_path() == log.resolve()
    trace("classify.verdict", on_topic=True)
    trace("retrieval.chunk", text="alpha")
    disable_trace()  # closes the file
    assert current_log_path() is None

    content = log.read_text(encoding="utf-8")
    assert "ChatQEC trace @" in content  # session header
    assert "[001]" in content and "[002]" in content  # numbered records
    assert "classify.verdict" in content
    assert "on_topic = True" in content
    assert "text = alpha" in content
    # The "[trace] " prefix is dropped in the file (index+timestamp replace it).
    assert "[trace]" not in content


def test_log_file_takes_precedence_over_stderr(tmp_path):
    captured: list[str] = []
    log = tmp_path / "trace.log"
    # Even with to_stderr=True, a log_file routes records to the file; an extra
    # in-memory sink still receives them.
    enable_trace(sink=captured.append, to_stderr=True, log_file=log)
    trace("step.one", k="v")
    disable_trace()
    assert captured and captured[0].startswith("[trace] step.one")
    assert "step.one" in log.read_text(encoding="utf-8")
