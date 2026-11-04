"""Benchmark adapter interface: turn an external QEC benchmark into scored chatQEC prompts.

An adapter owns two things: the items (a natural-language prompt plus the tool call a correct
system would make and the benchmark's gold numbers) and the scorer. Scoring is judge-free: it
reads the arguments and the JSON output of the tool calls the system made, so a number can only
score if a tool produced it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from chatqec.api_types import Answer, ToolCall
from chatqec.eval.metrics import parse_tool_output


@dataclass(frozen=True)
class BenchItem:
    id: str
    prompt: str
    expected_tool: str
    expected_args: dict[str, Any]
    gold: dict[str, float]
    meta: dict[str, Any] = field(default_factory=dict)


class Benchmark(ABC):
    """One external benchmark. Subclasses register with ``@register_benchmark(name)``."""

    name: str = ""

    @abstractmethod
    def items(self) -> list[BenchItem]:
        """Every item, in a stable order."""

    @abstractmethod
    def score(self, item: BenchItem, answer: Answer) -> dict[str, Any]:
        """Judge-free score for one answer; must include a boolean ``passed``."""


def tool_output(call: ToolCall) -> dict[str, Any]:
    """The tool's output as a dict (JSON or the Python repr chatQEC records), else {}."""
    return parse_tool_output(call.output)


def last_call(answer: Answer, tool: str, **where: Any) -> ToolCall | None:
    """The last call to ``tool``, preferring one whose arguments match ``where``."""
    calls = [c for c in answer.tool_calls if c.name == tool]
    matching = [c for c in calls if all(c.input.get(k) == v for k, v in where.items())]
    pool = matching or calls
    return pool[-1] if pool else None
