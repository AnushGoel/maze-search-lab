"""Heuristic functions for A* and greedy best-first search.

Every heuristic has the signature ``h(cell, goal) -> number`` expected by
:func:`mazesearch.core.a_star_search`. Besides the usual distance
estimates, the module implements the landmark ("ALT") heuristic of
Goldberg and Harrelson (2005): exact distances to a few landmark cells,
combined through the triangle inequality, give lower bounds that follow
the corridors of a maze instead of the straight line.
"""

from __future__ import annotations

import math
import time
from array import array
from collections import deque
from typing import List, Optional, Sequence, Tuple

from .core import WALL, Cell, Heuristic, Maze, manhattan_distance

__all__ = [
    "LandmarkHeuristic", "distance_table", "euclidean_distance",
    "manhattan_distance", "weighted", "zero_heuristic",
]


def zero_heuristic(cell: Cell, goal: Cell) -> int:
    """Return 0 for every cell, which turns A* into uniform-cost search.

    Args:
        cell: Ignored.
        goal: Ignored.

    Returns:
        Always 0.
    """
    return 0


def euclidean_distance(cell: Cell, goal: Cell) -> float:
    """Return the straight-line distance between two cells.

    It is admissible on a 4-connected grid but never larger than the
    Manhattan distance, so it is the less informed of the two.
    """
    return math.hypot(cell[0] - goal[0], cell[1] - goal[1])


def weighted(heuristic: Heuristic, weight: float) -> Heuristic:
    """Return ``heuristic`` multiplied by ``weight`` (Pohl, 1970).

    A weight above 1 makes the search greedier; the path it returns may
    then be longer than the shortest one.
    """
    def scaled(cell: Cell, goal: Cell) -> float:
        return weight * heuristic(cell, goal)

    scaled.__name__ = f"weighted_{weight:g}"
    return scaled


def distance_table(maze: Maze, source: Cell) -> "array[int]":
    """Exact move distances from ``source`` to every cell, by BFS.

    Args:
        maze: The maze to measure.
        source: The cell that distances are measured from.

    Returns:
        A flat ``array('i')`` indexed by ``row * width + col``. Walls and
        unreachable cells hold -1. At four bytes per cell the table needs
        about a tenth of the memory of a dictionary keyed by cells.
    """
    width, height = maze.width, maze.height
    flat = "".join(maze.rows)
    dist = array("i", [-1]) * (width * height)
    if not maze.is_open(source):
        return dist
    first = source[0] * width + source[1]
    dist[first] = 0
    queue = deque([first])
    last_row = (height - 1) * width
    while queue:
        index = queue.popleft()
        step = dist[index] + 1
        col = index % width
        for nxt, inside in ((index - width, index >= width),
                            (index + width, index < last_row),
                            (index - 1, col > 0),
                            (index + 1, col < width - 1)):
            if inside and dist[nxt] < 0 and flat[nxt] != WALL:
                dist[nxt] = step
                queue.append(nxt)
    return dist


class LandmarkHeuristic:
    """ALT lower bound: the largest |d(L, n) - d(L, goal)| over landmarks.

    For every landmark L, the triangle inequality on an undirected graph
    gives d(n, goal) >= |d(L, n) - d(L, goal)|. The maximum over a set of
    landmarks therefore never overestimates (admissible) and changes by
    at most one per move (consistent). Each landmark costs one complete
    breadth-first search to prepare, an investment that pays off when
    many queries are answered on the same maze.

    Attributes:
        landmarks: The landmark cells, in the order they were chosen.
        seconds_per_landmark: Time spent preparing each landmark.
    """

    def __init__(self, width: int, landmarks: Sequence[Cell],
                 tables: Sequence["array[int]"],
                 seconds_per_landmark: Sequence[float] = ()) -> None:
        """Wrap precomputed tables; see :meth:`farthest`."""
        self.landmarks: List[Cell] = list(landmarks)
        self.seconds_per_landmark: List[float] = list(seconds_per_landmark)
        self._width = width
        self._tables = list(tables)
        self._goal: Optional[Cell] = None
        self._active: List[Tuple["array[int]", int]] = []

    @property
    def build_seconds(self) -> float:
        """Total time spent preparing the landmark tables."""
        return sum(self.seconds_per_landmark)

    @classmethod
    def from_cells(cls, maze: Maze,
                   cells: Sequence[Cell]) -> "LandmarkHeuristic":
        """Prepare tables for explicitly chosen landmark cells."""
        tables, seconds = [], []
        for cell in cells:
            started = time.perf_counter()
            tables.append(distance_table(maze, cell))
            seconds.append(time.perf_counter() - started)
        return cls(maze.width, cells, tables, seconds)

    @classmethod
    def farthest(cls, maze: Maze, count: int,
                 seed_cell: Optional[Cell] = None) -> "LandmarkHeuristic":
        """Choose ``count`` landmarks by farthest-point selection.

        The first landmark is the open cell farthest from ``seed_cell``
        (the start by default). Each later landmark is the cell farthest
        from all landmarks chosen so far, which spreads them towards the
        far reaches of the maze (Goldberg & Harrelson, 2005).
        """
        started = time.perf_counter()
        scores = distance_table(maze, seed_cell or maze.start)
        nearest = None
        landmarks, tables, seconds = [], [], []
        for _ in range(count):
            best = max(range(len(scores)), key=scores.__getitem__)
            if scores[best] <= 0:
                break  # Every reachable cell is already a landmark.
            cell = divmod(best, maze.width)
            table = distance_table(maze, cell)
            nearest = (table if nearest is None
                       else array("i", map(min, nearest, table)))
            scores = nearest
            landmarks.append(cell)
            tables.append(table)
            seconds.append(time.perf_counter() - started)
            started = time.perf_counter()
        return cls(maze.width, landmarks, tables, seconds)

    def subset(self, count: int) -> "LandmarkHeuristic":
        """Return a heuristic that uses only the first ``count`` landmarks."""
        return LandmarkHeuristic(self._width, self.landmarks[:count],
                                 self._tables[:count],
                                 self.seconds_per_landmark[:count])

    def _prepare(self, goal: Cell) -> None:
        """Cache each landmark's distance to ``goal``."""
        index = goal[0] * self._width + goal[1]
        self._active = [(table, table[index]) for table in self._tables
                        if table[index] >= 0]
        self._goal = goal

    def __call__(self, cell: Cell, goal: Cell) -> int:
        """Return the landmark lower bound on the moves from cell to goal."""
        if goal != self._goal:
            self._prepare(goal)
        index = cell[0] * self._width + cell[1]
        best = 0
        for table, to_goal in self._active:
            gap = table[index] - to_goal
            if gap < 0:
                gap = -gap
            if gap > best:
                best = gap
        return best
