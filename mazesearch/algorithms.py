"""More search algorithms, plus a registry that describes all of them.

The three classic searches live in :mod:`mazesearch.core`. This module
adds the variants used in the experiments: uniform-cost search, weighted
A*, greedy best-first search, bidirectional breadth-first search and A*
with landmark (ALT) lower bounds. Every function has the shape of the
core searches: it takes a :class:`~mazesearch.core.Maze` and an optional
``on_expand`` callback, and it returns a
:class:`~mazesearch.core.SearchResult`.
"""

from __future__ import annotations

import dataclasses
import heapq
import itertools
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .core import (Cell, ExpandHook, Heuristic, Maze, SearchResult,
                   a_star_search, breadth_first_search, depth_first_search,
                   manhattan_distance, reconstruct_path)
from .heuristics import LandmarkHeuristic, weighted, zero_heuristic


def uniform_cost_search(maze: Maze,
                        on_expand: ExpandHook = None) -> SearchResult:
    """Uniform-cost search: A* with h = 0.

    With unit move costs it expands cells in the same rings as BFS but
    pays for a priority queue; it is Dijkstra's (1959) algorithm stopped
    at the goal.
    """
    result = a_star_search(maze, zero_heuristic, on_expand=on_expand)
    return dataclasses.replace(result, algorithm="UCS")


def weighted_a_star_search(maze: Maze, weight: float = 2.0,
                           on_expand: ExpandHook = None) -> SearchResult:
    """A* with the Manhattan distance multiplied by ``weight``."""
    result = a_star_search(maze, weighted(manhattan_distance, weight),
                           on_expand=on_expand)
    return dataclasses.replace(result, algorithm=f"WA* (w={weight:g})")


def greedy_best_first_search(maze: Maze,
                             heuristic: Heuristic = manhattan_distance,
                             on_expand: ExpandHook = None) -> SearchResult:
    """Greedy best-first search: always expand the cell with smallest h.

    It ignores the cost already paid, so it is quick when the heuristic
    points the right way and takes long detours when it does not. Every
    reached cell is recorded, so it never loops and is complete on a
    finite maze, but it is not optimal.
    """
    start, goal = maze.start, maze.goal
    came_from: Dict[Cell, Optional[Cell]] = {start: None}
    order = itertools.count()
    frontier = [(heuristic(start, goal), next(order), start)]
    expanded, generated, max_frontier = 0, 1, 1
    while frontier:
        _, _, cell = heapq.heappop(frontier)
        if cell == goal:
            return SearchResult("Greedy", reconstruct_path(came_from, goal),
                                expanded, generated, max_frontier,
                                len(came_from))
        expanded += 1
        if on_expand is not None:
            on_expand(cell)
        for neighbor in maze.neighbors(cell):
            if neighbor not in came_from:
                came_from[neighbor] = cell
                heapq.heappush(frontier, (heuristic(neighbor, goal),
                                          next(order), neighbor))
                generated += 1
        if len(frontier) > max_frontier:
            max_frontier = len(frontier)
    return SearchResult("Greedy", None, expanded, generated, max_frontier,
                        len(came_from))


def bidirectional_bfs(maze: Maze,
                      on_expand: ExpandHook = None) -> SearchResult:
    """Breadth-first search from both ends at once, meeting in the middle.

    The side with the smaller frontier expands one complete layer at a
    time. When a layer reaches a cell that the other side has reached,
    the search finishes that layer, keeps the shortest connection found
    and stops, which keeps the result optimal for unit move costs.
    """
    start, goal = maze.start, maze.goal
    if start == goal:
        return SearchResult("Bi-BFS", [start], 0, 1, 1, 1)
    parents: List[Dict[Cell, Optional[Cell]]] = [{start: None},
                                                 {goal: None}]
    depth: List[Dict[Cell, int]] = [{start: 0}, {goal: 0}]
    layers: List[List[Cell]] = [[start], [goal]]
    expanded, generated, max_frontier = 0, 2, 2
    best, meeting = math.inf, None
    while layers[0] and layers[1] and meeting is None:
        side = 0 if len(layers[0]) <= len(layers[1]) else 1
        mine, theirs = parents[side], parents[1 - side]
        my_depth, their_depth = depth[side], depth[1 - side]
        next_layer: List[Cell] = []
        for cell in layers[side]:
            expanded += 1
            if on_expand is not None:
                on_expand(cell)
            for neighbor in maze.neighbors(cell):
                if neighbor in mine:
                    continue
                mine[neighbor] = cell
                my_depth[neighbor] = my_depth[cell] + 1
                generated += 1
                if neighbor in theirs:
                    total = my_depth[neighbor] + their_depth[neighbor]
                    if total < best:
                        best, meeting = total, neighbor
                next_layer.append(neighbor)
        layers[side] = next_layer
        max_frontier = max(max_frontier, len(layers[0]) + len(layers[1]))
    reached = len(parents[0]) + len(parents[1])
    if meeting is None:
        return SearchResult("Bi-BFS", None, expanded, generated,
                            max_frontier, reached)
    path = reconstruct_path(parents[0], meeting)
    cell = parents[1][meeting]
    while cell is not None:
        path.append(cell)
        cell = parents[1][cell]
    return SearchResult("Bi-BFS", path, expanded, generated, max_frontier,
                        reached)


def landmark_a_star_search(maze: Maze, landmarks: int = 8,
                           on_expand: ExpandHook = None,
                           heuristic: Optional[LandmarkHeuristic] = None
                           ) -> SearchResult:
    """A* guided by landmark (ALT) lower bounds.

    Args:
        maze: The maze to search.
        landmarks: How many landmarks to choose when ``heuristic`` is not
            given (farthest-point selection).
        on_expand: Optional callback invoked with every expanded cell.
        heuristic: A prepared :class:`LandmarkHeuristic` to reuse across
            queries. When omitted, preparing it is part of this call.

    Returns:
        The search result; the path is optimal because the bound is
        admissible.
    """
    if heuristic is None:
        heuristic = LandmarkHeuristic.farthest(maze, landmarks)
    result = a_star_search(maze, heuristic, on_expand=on_expand)
    label = f"A* + ALT (k={len(heuristic.landmarks)})"
    return dataclasses.replace(result, algorithm=label)


@dataclass(frozen=True)
class Algorithm:
    """Description of one search algorithm.

    Attributes:
        key: Short identifier used on the command line and in the app.
        label: Display name; matches ``SearchResult.algorithm``.
        family: ``"blind"`` or ``"informed"``.
        optimal: Whether the returned path is always a shortest one.
        search: The search function.
        summary: One-line description.
    """

    key: str
    label: str
    family: str
    optimal: bool
    search: Callable[..., SearchResult]
    summary: str


ALGORITHMS: Dict[str, Algorithm] = {spec.key: spec for spec in (
    Algorithm("bfs", "BFS", "blind", True, breadth_first_search,
              "FIFO queue: explores in rings of distance from S."),
    Algorithm("dfs", "DFS", "blind", False, depth_first_search,
              "LIFO stack: dives deep and backs up at dead ends."),
    Algorithm("bibfs", "Bi-BFS", "blind", True, bidirectional_bfs,
              "Two BFS waves, from S and from E, that meet."),
    Algorithm("ucs", "UCS", "blind", True, uniform_cost_search,
              "Priority queue on g(n); Dijkstra's algorithm."),
    Algorithm("astar", "A*", "informed", True, a_star_search,
              "Priority queue on g(n) + Manhattan distance."),
    Algorithm("wastar", "WA* (w=2)", "informed", False,
              weighted_a_star_search,
              "A* with 2 x Manhattan: greedier, may detour."),
    Algorithm("greedy", "Greedy", "informed", False,
              greedy_best_first_search,
              "Priority queue on h(n) alone."),
    Algorithm("alt", "A* + ALT (k=8)", "informed", True,
              landmark_a_star_search,
              "A* with landmark bounds (preparation included)."),
)}
