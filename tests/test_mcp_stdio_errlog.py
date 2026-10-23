"""_errlog: the MCP server subprocess needs a stderr with a real file descriptor."""


def test_errlog_falls_back_when_stderr_has_no_fileno(monkeypatch):
    """Jupyter kernels swap sys.stderr for a stream without fileno(); spawning must still work."""
    import io
    import sys

    from chatqec.conversation.mcp_stdio import _errlog

    monkeypatch.setattr(sys, "stderr", io.StringIO())
    stream = _errlog()
    assert stream is not sys.stderr
    stream.fileno()  # usable as a subprocess stderr


def test_errlog_uses_stderr_when_it_is_a_real_file(monkeypatch, tmp_path):
    import sys

    from chatqec.conversation.mcp_stdio import _errlog

    with open(tmp_path / "err.log", "w") as real:
        monkeypatch.setattr(sys, "stderr", real)
        assert _errlog() is real
