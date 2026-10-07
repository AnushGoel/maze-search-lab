"""Measurements that explain why the searches behave as they do.

* :func:`maze_stats` describes a maze's structure: dead ends, junctions,
  loops (the cyclomatic number) and tortuosity.
* :func:`heuristic_profile` counts the cells A* must expand for a given
  heuristic, using the classic condition f(n) = g*(n) + h(n) < C*
  (Hart et al., 1968; Dechter & Pearl, 1985).
* :func:`exit_census` finds, for every possible exit at once, how many
  cells BFS and DFS expand before reaching it.
"""

from __future__ import annotations

import itertools
from array import array
from typing import Callable, Dict, List, Optional, Tuple

from .core import (WALL, Cell, Heuristic, Maze, SearchResult,
                   breadth_first_search, depth_first_search,
                   manhattan_distance)
from .heuristics import distance_table

__all__ = ["exit_census", "expansion_order", "heuristic_profile",
           "maze_stats", "open_cells"]


def open_cells(maze: Maze) -> List[Cell]:
    """Every cell that is not a wall, in row-major order."""
    return [(r, c) for r, row in enumerate(maze.rows)
            for c, char in enumerate(row) if char != WALL]


def _count_regions(maze: Maze, cells: List[Cell]) -> int:
    """Number of connected regions formed by the open ``cells``."""
    seen = set()
    regions = 0
    for cell in cells:
        if cell in seen:
            continue
        regions += 1
        seen.add(cell)
        stack = [cell]
        while stack:
            for neighbor in maze.neighbors(stack.pop()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
    return regions


def maze_stats(maze: Maze) -> Dict[str, Optional[float]]:
    """Summarise the structure of a maze.

    Returns:
        A dictionary with the grid size, the number of open cells, dead
        ends (one open neighbor) and junctions (three or more), the
        number of independent loops (edges - cells + regions), the
        shortest route length from S to E (``None`` if unreachable), the
        Manhattan distance between S and E, and the tortuosity (route
        length divided by Manhattan distance).
    """
    cells = open_cells(maze)
    degrees = [sum(1 for _ in maze.neighbors(cell)) for cell in cells]
    regions = _count_regions(maze, cells)
    distances = distance_table(maze, maze.start)
    shortest = distances[maze.goal[0] * maze.width + maze.goal[1]]
    straight = manhattan_distance(maze.start, maze.goal)
    route = None if shortest < 0 else shortest
    return {
        "width": maze.width,
        "height": maze.height,
        "open_cells": len(cells),
        "dead_ends": sum(degree == 1 for degree in degrees),
        "junctions": sum(degree >= 3 for degree in degrees),
        "loops": sum(degrees) // 2 - len(cells) + regions,
        "regions": regions,
        "shortest_path": route,
        "manhattan": straight,
        "tortuosity": None if route is None else route / straight,
    }


def expansion_order(maze: Maze, search: Callable[..., SearchResult],
                    **options: object) -> Tuple["array[int]", SearchResult]:
    """Run ``search`` and record the step at which each cell is expanded.

    Returns:
        A flat ``array('i')`` (index ``row * width + col``) holding each
        cell's expansion step, -1 where the cell was never expanded, and
        the search result.
    """
    order = array("i", [-1]) * (maze.width * maze.height)
    width = maze.width
    step = itertools.count()

    def record(cell: Cell) -> None:
        order[cell[0] * width + cell[1]] = next(step)

    result = search(maze, on_expand=record, **options)
    return order, result


def heuristic_profile(maze: Maze,
                      heuristic: Heuristic = manhattan_distance
                      ) -> Dict[str, int]:
    """Count the cells A* must, may and need not expand.

    With a consistent heuristic, A* expands every cell whose
    f(n) = g*(n) + h(n) is below the optimal cost C*, may expand cells
    with f(n) = C*, and never expands cells with f(n) > C*. BFS, by
    contrast, expands (almost) every cell closer to S than E is. The
    cells in between, ``prunable``, are the only ones a heuristic can
    save.

    Raises:
        ValueError: If the exit cannot be reached.
    """
    g_star = distance_table(maze, maze.start)
    width, goal = maze.width, maze.goal
    c_star = g_star[goal[0] * width + goal[1]]
    if c_star < 0:
        raise ValueError("the exit cannot be reached from the start")
    reachable = closer = must = may = h_max = 0
    for index, cost in enumerate(g_star):
        if cost < 0:
            continue
        reachable += 1
        h = heuristic(divmod(index, width), goal)
        h_max = max(h_max, h)
        if cost < c_star:
            closer += 1
        if cost + h < c_star:
            must += 1
        if cost + h <= c_star:
            may += 1
    return {"c_star": c_star, "h_max": int(h_max), "reachable": reachable,
            "closer_than_exit": closer, "must_expand": must,
            "may_expand": may, "prunable": closer - must}


def exit_census(maze: Maze) -> Tuple["array[int]", "array[int]"]:
    """Cells BFS and DFS expand before reaching each possible exit.

    A blind search visits cells in an order that does not depend on the
    goal: the goal only decides when it stops. One complete traversal
    therefore yields the cost of every possible exit at once. DFS reaches
    exit g after expanding exactly the cells it expands before g; BFS
    (with its goal test on generation) reaches g while expanding g's
    earliest-expanded neighbor.

    Returns:
        Two flat arrays, for BFS and DFS, indexed like
        :func:`~mazesearch.heuristics.distance_table`; -1 marks walls,
        unreachable cells and the start.
    """
    no_exit = Maze(maze.rows, maze.start, (-1, -1))
    bfs_order, _ = expansion_order(no_exit, breadth_first_search)
    dfs_order, _ = expansion_order(no_exit, depth_first_search)
    width = maze.width
    bfs_cost = array("i", [-1]) * len(bfs_order)
    dfs_cost = array("i", [-1]) * len(dfs_order)
    start_index = maze.start[0] * width + maze.start[1]
    for index, rank in enumerate(dfs_order):
        if rank < 0 or index == start_index:
            continue
        dfs_cost[index] = rank
        cell = divmod(index, width)
        bfs_cost[index] = 1 + min(
            bfs_order[n[0] * width + n[1]] for n in maze.neighbors(cell)
            if bfs_order[n[0] * width + n[1]] >= 0)
    return bfs_cost, dfs_cost
