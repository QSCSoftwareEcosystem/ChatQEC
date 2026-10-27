"""Source gate for guppy_run: decide whether a submitted program may be executed at all.

Why this is enough to be useful: a Guppy function body is compiled by Guppy, not run by CPython,
so CPython only executes the module's top-level statements, the decorator expressions, argument
defaults and annotations. This gate allows exactly:

  - imports from ``guppylang`` / ``guppylang.*`` (and ``from __future__ import annotations``)
  - docstrings
  - functions decorated with a bare ``@guppy``, with no defaults, no */** arguments and simple
    annotations, whose bodies use only a small allowlist of Guppy-compatible statements

Bodies are checked as well, because Guppy's ``py(...)`` and ``comptime`` evaluate arbitrary Python
at compile time; those names (and dunder access, ``exec``, ``open``, ``getattr`` ...) are refused
wherever they appear, including through an import alias.

This is a gate, not an OS sandbox. A bug in guppylang's own compiler is out of its reach, which is
why guppy_run is also off by default (see guppy_run.py).
"""

from __future__ import annotations

import ast

_DENIED_NAMES = frozenset(
    {
        "py", "comptime", "exec", "eval", "compile", "open", "getattr", "setattr", "delattr",
        "globals", "locals", "vars", "input", "breakpoint", "exit", "quit", "help", "importlib",
    }
)  # fmt: skip
_ALLOWED_BODY = (
    ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Return, ast.If, ast.While, ast.For, ast.Break,
    ast.Continue, ast.Pass, ast.Expr, ast.Assert, ast.Call, ast.keyword, ast.BinOp, ast.UnaryOp,
    ast.BoolOp, ast.Compare, ast.IfExp, ast.Constant, ast.Name, ast.Load, ast.Store, ast.Subscript,
    ast.Slice, ast.Tuple, ast.List, ast.Attribute, ast.ListComp, ast.comprehension, ast.Starred,
    ast.operator, ast.unaryop, ast.boolop, ast.cmpop,
)  # fmt: skip
_ALLOWED_ANNOTATION = (
    ast.Name, ast.Subscript, ast.Tuple, ast.List, ast.Attribute, ast.Constant, ast.Load,
    ast.BinOp, ast.BitOr, ast.Slice,
)  # fmt: skip


def _is_guppylang(module: str | None) -> bool:
    return module == "guppylang" or (module or "").startswith("guppylang.")


def _check_identifier(name: str) -> None:
    if name.startswith("__") or name in _DENIED_NAMES:
        kind = "dunder names" if name.startswith("__") else f"'{name}'"
        raise ValueError(f"{kind} are not allowed in a Guppy program ({name})")


def _check_annotation(node: ast.expr | None) -> None:
    if node is None:
        return
    for sub in ast.walk(node):
        if not isinstance(sub, _ALLOWED_ANNOTATION):
            raise ValueError(
                f"only simple type names are allowed in an annotation, not {type(sub).__name__}"
            )
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            raise ValueError(
                "string (forward-reference) annotations are not allowed in an annotation"
            )
        if isinstance(sub, ast.Name):
            _check_identifier(sub.id)
        if isinstance(sub, ast.Attribute):
            _check_identifier(sub.attr)


def _check_import(node: ast.Import | ast.ImportFrom) -> None:
    if isinstance(node, ast.ImportFrom):
        if node.level == 0 and node.module == "__future__":
            if any(a.name != "annotations" for a in node.names):
                raise ValueError("only 'from __future__ import annotations' is allowed")
            return
        if node.level != 0 or not _is_guppylang(node.module):
            raise ValueError(f"only imports from guppylang are allowed, not from {node.module!r}")
        for alias in node.names:
            if alias.name != "*":
                _check_identifier(alias.name)
        return
    for alias in node.names:
        if not _is_guppylang(alias.name):
            raise ValueError(f"only imports of guppylang are allowed, not import {alias.name!r}")
        if alias.asname:
            _check_identifier(alias.asname)


def _check_function(node: ast.FunctionDef) -> None:
    if len(node.decorator_list) != 1 or not (
        isinstance(node.decorator_list[0], ast.Name) and node.decorator_list[0].id == "guppy"
    ):
        raise ValueError(f"function {node.name!r} must have exactly the decorator @guppy")
    a = node.args
    if a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs:
        raise ValueError(
            "*args, **kwargs, keyword-only and positional-only arguments are not supported"
        )
    if a.defaults or any(d is not None for d in a.kw_defaults):
        raise ValueError("default argument values are not allowed")
    if getattr(node, "type_params", None):
        raise ValueError("type parameters are not allowed")
    for arg in a.args:
        _check_identifier(arg.arg)
        _check_annotation(arg.annotation)
    _check_annotation(node.returns)
    for stmt in node.body:
        for sub in ast.walk(stmt):
            if not isinstance(sub, _ALLOWED_BODY):
                raise ValueError(f"{type(sub).__name__} is not allowed in a Guppy function")
            if isinstance(sub, ast.Name):
                _check_identifier(sub.id)
            elif isinstance(sub, ast.Attribute):
                _check_identifier(sub.attr)


def validate_guppy_source(source: str) -> None:
    """Raise ValueError, naming the reason, unless ``source`` is a plain Guppy program."""
    try:
        tree = ast.parse(source, mode="exec")
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise ValueError(f"Could not parse the program: {type(exc).__name__}: {exc}") from exc
    functions: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            _check_import(node)
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue  # a docstring
        elif isinstance(node, ast.FunctionDef):
            _check_function(node)
            functions.add(node.name)
        elif isinstance(node, ast.AsyncFunctionDef):
            raise ValueError("async functions are not allowed")
        else:
            raise ValueError(
                f"{type(node).__name__} is not allowed at the top level; "
                "use imports from guppylang and @guppy functions only"
            )
    if "main" not in functions:
        raise ValueError("define a function named main (the entry point to run)")
