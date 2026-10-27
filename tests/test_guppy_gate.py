"""The source gate for guppy_run: only guppylang imports and @guppy functions may run as Python.

Guppy function BODIES are compiled by Guppy, not run by CPython, so CPython only ever executes
the top-level statements, decorators, defaults and annotations. The exception is Guppy's
`py(...)` / `comptime`, which evaluate Python at compile time, so bodies are checked too.
"""

import pytest

from chatqec_mcp_tools.tools._guppy_gate import validate_guppy_source

HEADER = (
    "from guppylang import guppy\n"
    "from guppylang.std.builtins import result\n"
    "from guppylang.std.quantum import qubit, h, cx, measure\n\n"
)
BELL = (
    HEADER
    + "@guppy\ndef main() -> None:\n"
    + "    q0 = qubit()\n    q1 = qubit()\n    h(q0)\n    cx(q0, q1)\n"
    + '    result("c0", measure(q0))\n    result("c1", measure(q1))\n'
)


def _prog(body: str, header: str = HEADER, sig: str = "def main() -> None:") -> str:
    indented = "".join(f"    {line}\n" for line in body.splitlines())
    return f"{header}@guppy\n{sig}\n{indented}"


def test_bell_program_passes():
    validate_guppy_source(BELL)


def test_helper_functions_loops_and_conditionals_pass():
    src = (
        HEADER
        + "@guppy\ndef parity(a: bool, b: bool) -> bool:\n    return a != b\n\n"
        + "@guppy\ndef main() -> None:\n    q = qubit()\n    h(q)\n"
        + "    n = 0\n    for i in range(3):\n        if parity(True, False):\n"
        + '            n += 1\n    result("c", measure(q))\n'
    )
    validate_guppy_source(src)


@pytest.mark.parametrize(
    "label,source,reason",
    [
        ("import os", "import os\n" + BELL, "import"),
        ("from os import", "from os import system\n" + BELL, "import"),
        ("guppylang_internals", "from guppylang_internals import x\n" + BELL, "import"),
        (
            "top-level call",
            HEADER + "open('/etc/passwd')\n@guppy\ndef main() -> None:\n    pass\n",
            "top level",
        ),
        (
            "top-level assign",
            HEADER + "x = __import__('os')\n@guppy\ndef main() -> None:\n    pass\n",
            "top level",
        ),
        (
            "class def",
            HEADER + "class A:\n    pass\n@guppy\ndef main() -> None:\n    pass\n",
            "top level",
        ),
        ("py() in body", _prog('result("x", py(__import__("os").getcwd()))'), "py"),
        (
            "star import then py",
            "from guppylang.std.builtins import *\n"
            + _prog("x = py(1)", header="@guppy\n"[:0] + "from guppylang import guppy\n"),
            "py",
        ),
        ("alias of py", "from guppylang.std.builtins import py as p\n" + BELL, "py"),
        (
            "attribute py",
            "import guppylang\n"
            + _prog(
                "x = guppylang.py(1)", header="from guppylang import guppy\nimport guppylang\n"
            ),
            "py",
        ),
        (
            "comptime decorator",
            HEADER + "@guppy.comptime\ndef main() -> None:\n    pass\n",
            "decorator",
        ),
        (
            "decorator call",
            HEADER + "@guppy(open('x'))\ndef main() -> None:\n    pass\n",
            "decorator",
        ),
        (
            "default argument",
            HEADER
            + "@guppy\ndef helper(x: int = open('f')) -> int:\n    return x\n@guppy\ndef main() -> None:\n    pass\n",
            "default",
        ),
        (
            "call in annotation",
            HEADER
            + "@guppy\ndef helper(x: __import__('os').system('true')) -> int:\n    return 1\n@guppy\ndef main() -> None:\n    pass\n",
            "annotation",
        ),
        (
            "string annotation",
            HEADER
            + '@guppy\ndef helper(x: "py(1)") -> int:\n    return 1\n@guppy\ndef main() -> None:\n    pass\n',
            "annotation",
        ),
        (
            "string return annotation",
            HEADER
            + "@guppy\ndef helper() -> \"__import__('os')\":\n    return 1\n@guppy\ndef main() -> None:\n    pass\n",
            "annotation",
        ),
        ("dunder attribute", _prog("x = ().__class__"), "dunder"),
        ("dunder name", _prog("x = __builtins__"), "dunder"),
        ("exec in body", _prog('exec("1")'), "exec"),
        ("getattr in body", _prog('x = getattr(1, "real")'), "getattr"),
        ("lambda", _prog("f = lambda: 1"), "Lambda"),
        ("with statement", _prog("with open('x') as f:\n    pass"), "With"),
        ("global statement", _prog("global x"), "Global"),
        ("nested def", _prog("def inner() -> None:\n    pass"), "FunctionDef"),
        ("f-string", _prog('x = f"{1}"'), "JoinedStr"),
        ("async def", HEADER + "@guppy\nasync def main() -> None:\n    pass\n", "async"),
        ("no main", HEADER + "@guppy\ndef other() -> None:\n    pass\n", "main"),
        ("syntax error", "def (:\n", "parse"),
        ("varargs", HEADER + "@guppy\ndef main(*a) -> None:\n    pass\n", "argument"),
    ],
)
def test_hostile_or_unsupported_sources_are_rejected(label, source, reason):
    with pytest.raises(ValueError) as exc:
        validate_guppy_source(source)
    assert reason.lower() in str(exc.value).lower(), f"{label}: {exc.value}"


def test_deeply_nested_source_is_rejected_not_a_crash():
    src = HEADER + "@guppy\ndef main() -> None:\n    x = " + "(" * 5000 + "1" + ")" * 5000 + "\n"
    with pytest.raises(ValueError):
        validate_guppy_source(src)
