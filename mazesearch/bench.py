"""Dependency-free helpers for timing and memory measurements."""

from __future__ import annotations

import gc
import statistics
import time
import tracemalloc
from typing import Callable, Dict, List, Sequence

from .core import Maze, SearchResult

Search = Callable[[Maze], SearchResult]

__all__ = ["interleaved_times", "peak_memory_mib", "summarize"]


def interleaved_times(maze: Maze, searches: Dict[str, Search],
                      repeats: int = 7,
                      warmup: int = 1) -> Dict[str, List[float]]:
    """Time each search on ``maze``, interleaving runs round-robin.

    Each search first gets ``warmup`` untimed runs. The timed runs then
    rotate through the searches (A, B, C, A, B, C, ...), so a change in
    machine load affects every algorithm alike, and garbage is collected
    before each run so no algorithm pays for another's allocations.
    """
    for _ in range(warmup):
        for search in searches.values():
            search(maze)
    times: Dict[str, List[float]] = {name: [] for name in searches}
    for _ in range(repeats):
        for name, search in searches.items():
            gc.collect()
            started = time.perf_counter()
            search(maze)
            times[name].append(time.perf_counter() - started)
    return times


def peak_memory_mib(search: Search, maze: Maze) -> float:
    """Peak memory (MiB) that Python allocates while ``search`` runs.

    Measured with :mod:`tracemalloc`, which slows execution, so never
    combine it with timing runs.
    """
    gc.collect()
    tracemalloc.start()
    try:
        search(maze)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / 2 ** 20


def summarize(samples: Sequence[float]) -> Dict[str, float]:
    """Median, quartiles (inclusive method), IQR, minimum and maximum."""
    data = sorted(samples)
    if len(data) > 1:
        q1, _, q3 = statistics.quantiles(data, n=4, method="inclusive")
    else:
        q1 = q3 = data[0]
    return {"median": statistics.median(data), "q1": q1, "q3": q3,
            "iqr": q3 - q1, "min": data[0], "max": data[-1],
            "n": float(len(data))}
