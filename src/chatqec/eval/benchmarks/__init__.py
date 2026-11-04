"""External QEC benchmarks for the chatQEC evaluation (Factory & Registry)."""

from __future__ import annotations

from collections.abc import Callable

from chatqec.eval.benchmarks.base import BenchItem, Benchmark

BENCHMARKS: dict[str, type[Benchmark]] = {}


def register_benchmark(name: str) -> Callable[[type[Benchmark]], type[Benchmark]]:
    def decorator(cls: type[Benchmark]) -> type[Benchmark]:
        cls.name = name
        BENCHMARKS[name] = cls
        return cls

    return decorator


def get_benchmark(name: str) -> type[Benchmark]:
    _load_adapters()
    if name not in BENCHMARKS:
        raise KeyError(f"unknown benchmark {name!r}; known: {sorted(BENCHMARKS)}")
    return BENCHMARKS[name]


def _load_adapters() -> None:
    # Importing an adapter registers it.
    from chatqec.eval.benchmarks import decoder_bench  # noqa: F401


__all__ = ["BENCHMARKS", "BenchItem", "Benchmark", "get_benchmark", "register_benchmark"]
