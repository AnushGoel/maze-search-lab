"""Core maze model and the three classic searches: BFS, DFS and A*.

A maze is a text grid in which ``S`` marks the start, ``E`` the exit and
``X`` a wall; every other character is open floor. An optional first
line ``"width height"`` gives the dimensions. Moves go up, down, left or
right and each costs one step.

The module provides:

* :class:`Maze` and :func:`load_maze`, which parse mazes from files
  (plain or gzip-compressed), text, lists of rows or character grids;
* :func:`breadth_first_search`, :func:`depth_first_search` and
  :func:`a_star_search`, which return a :class:`SearchResult` holding
  the path and the work counters used in the analysis;
* :func:`solve_maze` and the wrappers :func:`maze_solver_one` (BFS),
  :func:`maze_solver_two` (DFS) and :func:`maze_solver_three` (A*),
  which return the maze with the path marked ``*`` in the same form it
  was given, or ``None`` when no path exists.

Only the standard library is used and no search is recursive, so mazes
with millions of cells are solved without reaching Python's recursion
limit.

Example:
    >>> maze = "5 3\\nXXXXX\\nXS EX\\nXXXXX\\n"
    >>> print(maze_solver_three(maze), end="")
    5 3
    XXXXX
    XS*EX
    XXXXX
"""
from __future__ import annotations

import gzip
import heapq
import math
import os
import re
from collections import deque
from dataclasses import dataclass
from typing import (Any, Callable, Dict, Iterator, List, Optional, Sequence,
                    Set, Tuple)

# ---------------------------------------------------------------------------
# Constants and type aliases
# ---------------------------------------------------------------------------

WALL = "X"
START = "S"
GOAL = "E"
PATH_MARK = "*"

# Moves as (row offset, column offset) in the fixed order north, east,
# south, west. Only orthogonal moves are allowed and each costs 1.
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))

# The optional first line of a maze file: "<width> <height>".
_HEADER_PATTERN = re.compile(r"^\s*(\d+)\s+(\d+)\s*$")

Cell = Tuple[int, int]
Move = Tuple[int, int]
Heuristic = Callable[[Cell, Cell], float]
ExpandHook = Optional[Callable[[Cell], None]]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Maze:
    """A parsed maze: an immutable character grid and its key cells.

    Cells are ``(row, col)`` pairs counted from the top-left corner, so
    ``rows[row][col]`` is the character stored in a cell.

    Attributes:
        rows: The grid, one string per row, all of the same length.
        start: Position of the start cell ``S``.
        goal: Position of the exit cell ``E``.
    """

    rows: Tuple[str, ...]
    start: Cell
    goal: Cell

    @property
    def height(self) -> int:
        """Number of rows in the grid."""
        return len(self.rows)

    @property
    def width(self) -> int:
        """Number of columns in the grid."""
        return len(self.rows[0]) if self.rows else 0

    def is_open(self, cell: Cell) -> bool:
        """Return True if ``cell`` is inside the grid and not a wall."""
        row, col = cell
        return (0 <= row < self.height and 0 <= col < self.width
                and self.rows[row][col] != WALL)

    def neighbors(self, cell: Cell,
                  moves: Sequence[Move] = MOVES) -> Iterator[Cell]:
        """Yield the open cells reachable from ``cell`` in one move.

        Args:
            cell: The cell being expanded.
            moves: Move offsets to try, in the order they are yielded.

        Yields:
            Each open neighboring cell, in the order given by ``moves``.
        """
        rows = self.rows
        height, width = len(rows), len(rows[0])
        row, col = cell
        for d_row, d_col in moves:
            n_row, n_col = row + d_row, col + d_col
            if (0 <= n_row < height and 0 <= n_col < width
                    and rows[n_row][n_col] != WALL):
                yield n_row, n_col


@dataclass(frozen=True)
class SearchResult:
    """The outcome of one search, with the counters used in the report.

    Attributes:
        algorithm: Short name of the algorithm that produced the result.
        path: Cells from the start to the goal inclusive, or ``None``
            if the goal is unreachable.
        nodes_expanded: Cells taken off the frontier and expanded.
        nodes_generated: Insertions into the frontier, counting the
            start cell and any duplicate entries.
        max_frontier: Largest number of entries held in the frontier.
        nodes_reached: Distinct cells recorded in the search tree.
    """

    algorithm: str
    path: Optional[List[Cell]]
    nodes_expanded: int
    nodes_generated: int
    max_frontier: int
    nodes_reached: int

    @property
    def solved(self) -> bool:
        """True if a path from the start to the goal was found."""
        return self.path is not None

    @property
    def path_length(self) -> Optional[int]:
        """Number of moves on the path, or ``None`` if unsolved."""
        return None if self.path is None else len(self.path) - 1


def reconstruct_path(came_from: Dict[Cell, Optional[Cell]],
                     goal: Cell) -> List[Cell]:
    """Follow parent links back from ``goal`` and return the path.

    Args:
        came_from: Maps each reached cell to the cell it was reached
            from; the start cell maps to ``None``.
        goal: The last cell of the path.

    Returns:
        The cells from the start to ``goal``, inclusive.
    """
    path = []
    cell: Optional[Cell] = goal
    while cell is not None:
        path.append(cell)
        cell = came_from[cell]
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Blind search 1: breadth-first search
# ---------------------------------------------------------------------------


def breadth_first_search(maze: Maze,
                         on_expand: ExpandHook = None) -> SearchResult:
    """Find a shortest path with breadth-first search (blind search).

    The frontier is a FIFO queue, so cells are expanded in order of
    depth (number of moves from the start). Because all moves cost the
    same, the first time the goal is *generated* it has been reached by
    a shortest path; the goal test therefore runs when a cell is
    generated rather than when it is expanded, which saves a layer.

    Properties: complete; optimal for unit move costs; O(V + E) time
    and O(V) memory for a maze with V open cells and E connections.

    Args:
        maze: The maze to search.
        on_expand: Optional callback invoked with every expanded cell
            (used for visualization; it does not change the search).

    Returns:
        The search result. ``path`` is ``None`` if the goal is
        unreachable.
    """
    start, goal = maze.start, maze.goal
    came_from: Dict[Cell, Optional[Cell]] = {start: None}
    if start == goal:
        return SearchResult("BFS", [start], 0, 1, 1, 1)

    frontier = deque([start])
    expanded, generated, max_frontier = 0, 1, 1
    while frontier:
        cell = frontier.popleft()
        expanded += 1
        if on_expand is not None:
            on_expand(cell)
        for neighbor in maze.neighbors(cell):
            if neighbor in came_from:  # Already reached by a path as short.
                continue
            came_from[neighbor] = cell
            generated += 1
            if neighbor == goal:
                return SearchResult("BFS", reconstruct_path(came_from, goal),
                                    expanded, generated, max_frontier,
                                    len(came_from))
            frontier.append(neighbor)
        if len(frontier) > max_frontier:
            max_frontier = len(frontier)

    return SearchResult("BFS", None, expanded, generated, max_frontier,
                        len(came_from))


# ---------------------------------------------------------------------------
# Blind search 2: depth-first search
# ---------------------------------------------------------------------------


def depth_first_search(maze: Maze, moves: Sequence[Move] = MOVES,
                       on_expand: ExpandHook = None) -> SearchResult:
    """Find a path with depth-first search (blind search).

    The frontier is a LIFO stack, so the search always continues from
    the most recently generated cell and backs up only at dead ends. An
    explicit stack replaces recursion, which would exceed Python's
    default limit of 1,000 nested calls long before the end of a large
    maze. A cell is marked as explored when it is popped, reproducing
    the expansion order of recursive DFS; a cell may therefore sit on
    the stack more than once, and stale copies are skipped. Neighbors
    are pushed in reverse so that the first move in ``moves`` is tried
    first.

    Properties: complete on finite mazes (the explored set prevents
    cycles) but not optimal; O(V + E) time and O(V) memory in the
    worst case.

    Args:
        maze: The maze to search.
        moves: Order in which moves are tried; north, east, south, west
            by default.
        on_expand: Optional callback invoked with every expanded cell.

    Returns:
        The search result. ``path`` is ``None`` if the goal is
        unreachable.
    """
    start, goal = maze.start, maze.goal
    push_order = tuple(reversed(moves))
    came_from: Dict[Cell, Optional[Cell]] = {}
    frontier: List[Tuple[Cell, Optional[Cell]]] = [(start, None)]
    expanded, generated, max_frontier = 0, 1, 1

    while frontier:
        cell, parent = frontier.pop()
        if cell in came_from:  # Stale copy: explored via another branch.
            continue
        came_from[cell] = parent
        if cell == goal:
            return SearchResult("DFS", reconstruct_path(came_from, goal),
                                expanded, generated, max_frontier,
                                len(came_from))
        expanded += 1
        if on_expand is not None:
            on_expand(cell)
        for neighbor in maze.neighbors(cell, push_order):
            if neighbor not in came_from:
                frontier.append((neighbor, cell))
                generated += 1
        if len(frontier) > max_frontier:
            max_frontier = len(frontier)

    return SearchResult("DFS", None, expanded, generated, max_frontier,
                        len(came_from))


# ---------------------------------------------------------------------------
# Heuristic search: A* with the Manhattan-distance heuristic
# ---------------------------------------------------------------------------


def manhattan_distance(cell: Cell, goal: Cell) -> int:
    """Return the Manhattan (L1) distance between two cells.

    Any path between the cells needs at least this many orthogonal
    moves, so the estimate never exceeds the true cost (admissible).
    One move changes it by at most one, so h(n) <= 1 + h(n') holds for
    neighboring cells n and n' (consistent).

    Args:
        cell: The cell being evaluated.
        goal: The goal cell.

    Returns:
        ``|row difference| + |column difference|``.
    """
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


def a_star_search(maze: Maze, heuristic: Heuristic = manhattan_distance,
                  on_expand: ExpandHook = None) -> SearchResult:
    """Find a shortest path with A* search (heuristic search).

    A* expands the frontier cell with the lowest f(n) = g(n) + h(n),
    where g(n) is the number of moves from the start and h(n) is the
    heuristic estimate of the moves still needed. With an admissible
    and consistent heuristic, such as the default Manhattan distance,
    A* is complete and optimal and expands each cell at most once.

    Ties in f are broken in favor of the smaller h (the cell that is
    closer to the goal and deeper in the search), then by position, so
    results are deterministic. The binary-heap frontier uses lazy
    deletion: when a shorter route to a queued cell is found, a new
    entry is pushed and the outdated one is skipped when it is popped.

    Properties: complete and optimal with an admissible heuristic;
    O(E log V) time and O(V) memory in the worst case.

    Args:
        maze: The maze to search.
        heuristic: Estimate ``h(cell, goal)``; Manhattan distance by
            default.
        on_expand: Optional callback invoked with every expanded cell.

    Returns:
        The search result. ``path`` is ``None`` if the goal is
        unreachable.
    """
    start, goal = maze.start, maze.goal
    g_cost: Dict[Cell, int] = {start: 0}
    came_from: Dict[Cell, Optional[Cell]] = {start: None}
    h_start = heuristic(start, goal)
    frontier: List[Tuple[float, float, Cell]] = [(h_start, h_start, start)]
    closed: Set[Cell] = set()
    expanded, generated, max_frontier = 0, 1, 1

    while frontier:
        _, _, cell = heapq.heappop(frontier)
        if cell in closed:  # Stale entry superseded by a cheaper one.
            continue
        if cell == goal:
            return SearchResult("A*", reconstruct_path(came_from, goal),
                                expanded, generated, max_frontier,
                                len(came_from))
        closed.add(cell)
        expanded += 1
        if on_expand is not None:
            on_expand(cell)
        g_next = g_cost[cell] + 1
        for neighbor in maze.neighbors(cell):
            if neighbor in closed or g_next >= g_cost.get(neighbor, math.inf):
                continue
            g_cost[neighbor] = g_next
            came_from[neighbor] = cell
            h_neighbor = heuristic(neighbor, goal)
            heapq.heappush(frontier,
                           (g_next + h_neighbor, h_neighbor, neighbor))
            generated += 1
        if len(frontier) > max_frontier:
            max_frontier = len(frontier)

    return SearchResult("A*", None, expanded, generated, max_frontier,
                        len(came_from))


# ---------------------------------------------------------------------------
# Reading mazes and writing solutions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Layout:
    """How a maze was supplied, so its solution can be returned alike.

    Attributes:
        kind: ``"text"``, ``"lines"`` or ``"grid"``.
        header: The original ``"width height"`` line, or ``None``.
        line_end: ``"\\n"`` if the input lines (``"lines"``) or the text
            as a whole (``"text"``) ended with a newline, else ``""``.
    """

    kind: str
    header: Optional[str]
    line_end: str


def _read_source(source: Any) -> Tuple[List[str], str, str]:
    """Return the raw lines of a maze source, its kind and line ending.

    Raises:
        FileNotFoundError: If ``source`` names a file that does not
            exist.
        TypeError: If ``source`` is of an unsupported type.
    """
    if isinstance(source, bytes):
        source = source.decode("utf-8")
    if isinstance(source, os.PathLike) or (
            isinstance(source, str) and "\n" not in source):
        opener = gzip.open if str(source).endswith(".gz") else open
        with opener(source, "rt", encoding="utf-8") as handle:
            source = handle.read()
    elif hasattr(source, "read"):  # An open file object.
        source = source.read()

    if isinstance(source, str):
        line_end = "\n" if source.endswith("\n") else ""
        return source.splitlines(), "text", line_end
    if isinstance(source, (list, tuple)):
        if all(isinstance(line, str) for line in source):
            line_end = "\n" if source and source[0].endswith("\n") else ""
            return [line.rstrip("\r\n") for line in source], "lines", line_end
        return ["".join(row) for row in source], "grid", ""
    raise TypeError(f"unsupported maze type: {type(source).__name__}")


def _find_unique(rows: Sequence[str], symbol: str) -> Cell:
    """Return the only cell that holds ``symbol``.

    Raises:
        ValueError: If ``symbol`` occurs zero times or more than once.
    """
    count = sum(row.count(symbol) for row in rows)
    if count != 1:
        raise ValueError(
            f"a maze needs exactly one {symbol!r} cell, found {count}")
    for row_index, row in enumerate(rows):
        col_index = row.find(symbol)
        if col_index != -1:
            return row_index, col_index
    raise AssertionError("unreachable")  # pragma: no cover


def _parse(source: Any) -> Tuple[Maze, _Layout]:
    """Parse any supported maze form and remember how it was supplied."""
    if isinstance(source, Maze):
        header = f"{source.width} {source.height}"
        return source, _Layout("text", header, "\n")

    lines, kind, line_end = _read_source(source)
    header = None
    width = height = 0
    if lines and _HEADER_PATTERN.match(lines[0]):
        header = lines[0]
        width, height = (int(value) for value in header.split())
        lines = lines[1:]

    rows = list(lines)
    while rows and rows[-1] == "":  # Drop blank lines after the grid.
        rows.pop()
    # Restore rows or trailing spaces that an editor may have stripped.
    rows.extend([""] * (height - len(rows)))
    width = max([width] + [len(row) for row in rows])
    rows = [row.ljust(width) for row in rows]

    maze = Maze(tuple(rows), _find_unique(rows, START),
                _find_unique(rows, GOAL))
    return maze, _Layout(kind, header, line_end)


def load_maze(source: Any) -> Maze:
    """Parse a maze from any form accepted by the solver functions.

    Args:
        source: A file path, maze text, list of row strings or
            character grid (see the module docstring).

    Returns:
        The parsed maze.

    Raises:
        ValueError: If the maze lacks exactly one ``S`` and one ``E``.
    """
    return _parse(source)[0]


def mark_path(maze: Maze, path: Sequence[Cell]) -> List[str]:
    """Return the maze rows with the path between S and E marked ``*``.

    The start and exit cells keep their ``S`` and ``E`` characters.
    """
    grid = [list(row) for row in maze.rows]
    for row, col in path[1:-1]:
        grid[row][col] = PATH_MARK
    return ["".join(chars) for chars in grid]


def render_solution(maze: Maze, path: Sequence[Cell]) -> str:
    """Return a solved maze as text in the maze-file format."""
    lines = [f"{maze.width} {maze.height}"] + mark_path(maze, path)
    return "\n".join(lines) + "\n"


def maze_to_text(maze: Maze) -> str:
    """Return ``maze`` as text in the maze-file format, header included."""
    return "\n".join([f"{maze.width} {maze.height}", *maze.rows]) + "\n"


def _format(rows: List[str], layout: _Layout) -> Any:
    """Convert solved rows back into the caller's original form."""
    if layout.kind == "grid":
        return [list(row) for row in rows]
    lines = rows if layout.header is None else [layout.header] + rows
    if layout.kind == "lines":
        return [line + layout.line_end for line in lines]
    return "\n".join(lines) + layout.line_end


# ---------------------------------------------------------------------------
# High-level solver functions
# ---------------------------------------------------------------------------

SEARCH_ALGORITHMS: Dict[str, Callable[[Maze], SearchResult]] = {
    "bfs": breadth_first_search,
    "dfs": depth_first_search,
    "astar": a_star_search,
}


def solve_maze(maze: Any,
               search: Callable[[Maze], SearchResult] = a_star_search
               ) -> Any:
    """Solve a maze with the given search function.

    Args:
        maze: The maze in any supported form (see the module docstring).
        search: A search function such as :func:`a_star_search`.

    Returns:
        The solved maze, in the same form as ``maze``, with the path
        marked ``*``; or ``None`` if no path from S to E exists.

    Raises:
        ValueError: If the maze lacks exactly one ``S`` and one ``E``.
    """
    parsed, layout = _parse(maze)
    result = search(parsed)
    if result.path is None:
        return None
    return _format(mark_path(parsed, result.path), layout)


def maze_solver_one(maze: Any) -> Any:
    """Solve a maze with breadth-first search (blind search).

    Args:
        maze: The maze in any supported form (see the module docstring).

    Returns:
        The solved maze in the same form as ``maze``, with a shortest
        path marked ``*``; or ``None`` if the maze is unsolvable.
    """
    return solve_maze(maze, breadth_first_search)


def maze_solver_two(maze: Any) -> Any:
    """Solve a maze with depth-first search (blind search).

    Args:
        maze: The maze in any supported form (see the module docstring).

    Returns:
        The solved maze in the same form as ``maze``, with the path DFS
        found marked ``*`` (not necessarily the shortest); or ``None``
        if the maze is unsolvable.
    """
    return solve_maze(maze, depth_first_search)


def maze_solver_three(maze: Any) -> Any:
    """Solve a maze with A* search and the Manhattan heuristic.

    Args:
        maze: The maze in any supported form (see the module docstring).

    Returns:
        The solved maze in the same form as ``maze``, with a shortest
        path marked ``*``; or ``None`` if the maze is unsolvable.
    """
    return solve_maze(maze, a_star_search)


# Descriptive aliases for the numbered solver functions.
solve_with_bfs = maze_solver_one
solve_with_dfs = maze_solver_two
solve_with_astar = maze_solver_three
