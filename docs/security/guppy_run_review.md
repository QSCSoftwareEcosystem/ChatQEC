# guppy_run security review (2026-11-04)

`guppy_run` (tools repo `chatqec-mcp-tools`) executes a Guppy program submitted through chat on
the Selene emulator. It is off unless `CHATQEC_ALLOW_GUPPY_EXEC=1` is set for the MCP server.
This review answers one question: can a chat user, or text the model reads (a pasted paper, a
retrieved chunk), run code outside the intended Guppy program?

Reviewed at tools commit `57ea239` against the installed runtime (Python 3.13, guppylang 1.2.0,
selene-sim 0.3.7) by an independent reviewer agent; the gate bypass in finding 1 was then
re-checked by hand before fixing. Findings marked *confirmed* were reproduced with harmless
canaries only (files created for the test, a closed localhost port), never real user files.

## Verdict

| Deployment | Before the fixes below | After the fixes below |
|---|---|---|
| Public demo | Not safe: any user could run Python and shell as the server uid | **Still not safe.** No OS isolation; keep it off |
| Trusted single user writing their own programs | Unsafe as soon as the model reads untrusted text | Acceptable: no known bypass of the gate, plus a second barrier |

## Findings and status

| # | Severity | Finding | Status |
|---|---|---|---|
| 1 | Critical | **Gate bypass, confirmed.** A body annotation written as a string, `x: "array[int, comptime(PAYLOAD)]" = array(1)`, passed the source gate. Guppy parses string (delayed) annotations itself and evaluates `comptime(...)` with `eval` and full builtins. Confirmed: reading a file, reading the server's environment through `/proc/<ppid>/environ`, running a shell command, and starting a detached process that outlived the process-group kill. | **Fixed** in `422b8c7`: body annotations get the signature annotation check, string constants are allowed only as the `result(...)` label, and the child replaces `eval_comptime_expr` with a function that raises, so a future bypass fails at compile time. Tests: `tests/test_guppy_gate.py`, `tests/test_guppy_exec.py`. |
| 2 | High | **No containment, confirmed.** The child runs as the server uid with no filesystem, network or process-count isolation. Unprivileged user namespaces are blocked on this host (`unshare -n`, `unshare -rn` and `bwrap` all fail with EPERM), and firejail/nsjail are not installed. | **Open.** Landlock (ABI 7) works unprivileged here, verified with a ctypes probe: it denied a canary read and a TCP connect. Plan: apply Landlock plus a libseccomp socket filter in the child bootstrap, `PR_SET_DUMPABLE=0` in the server, and kill every descendant (cgroup scope or child subreaper). For a public demo, run the tool server in a container or VM, or as a separate uid that holds no secrets. |
| 3 | Medium | One `guppy_run` call blocks the whole MCP server for up to 55 s, because `call_tool` runs the handler on the event loop. | **Open, deliberately.** Moving handlers to threads would make tools that are not thread-safe (CUDA-Q, Gurobi) run concurrently, since all visitors of one app share one server. Needs a per-tool lock design first. |
| 4 | Medium | `stderr_tail` returns raw compiler tracebacks, which was the exfiltration channel in finding 1. | **Open.** Narrowing it helps but does not close it. |
| 5 | Low | Import aliases (`import result as __import__`) and helper names that shadow what the tool appends (`print`, `dict`, `_json`) were not checked. | **Fixed** in `422b8c7`. |
| 6 | Low | `sandbox.py` was unused, and its `RLIMIT_NPROC=32` broke threads, `import numpy` and subprocesses. | **Removed** in `2408bfd`. |

User code never runs in the server process: the server only parses the source (`ast.parse`) and
calls `importlib.util.find_spec`. All compilation happens in the child.

## How to enable it (trusted single user only)

```bash
CHATQEC_ALLOW_GUPPY_EXEC=1 CHATQEC_MCP_SERVER=chatqec-mcp-tools \
  uv run streamlit run chatqec_app/streamlit_app.py
```

Do not enable it on a deployment that other people can reach until finding 2 is closed.
