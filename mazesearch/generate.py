"""Seeded maze generators and maze transformations.

Mazes use the classic layout found throughout the project: corridor
cells sit at odd (row, col) coordinates, and the cells between them are
walls that a generator may knock down. Three generators with very
different textures are provided (Buck, 2015):

* ``backtracker``: recursive backtracker (randomised depth-first
  search). Long, winding corridors and few dead ends.
* ``prim``: randomised Prim's algorithm. Short corridors and many dead
  ends.
* ``binary_tree``: every cell opens north or east. A strong diagonal
  bias, so routes are nearly straight.

All three produce *perfect* mazes, with exactly one route between any
two cells; :func:`braid` then adds loops.
"""

from __future__ import annotations

import gzip
import random
from typing import Callable, Dict, List, Optional, Sequence

from .core import GOAL, START, WALL, Cell, Maze, maze_to_text

__all__ = [
    "GENERATORS", "braid", "generate_maze", "pick_cell",
    "removable_walls", "save_maze", "with_endpoints",
    "without_interior_walls",
]

GENERATORS = ("backtracker", "prim", "binary_tree")
_STEPS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_SPACE = ord(" ")
Grid = List[bytearray]


def _backtracker(grid: Grid, rows: int, cols: int,
                 rng: random.Random) -> None:
    """Carve a perfect maze with an iterative randomised DFS."""
    visited = bytearray(rows * cols)
    i, j = rng.randrange(rows), rng.randrange(cols)
    visited[i * cols + j] = 1
    grid[2 * i + 1][2 * j + 1] = _SPACE
    stack = [(i, j)]
    while stack:
        i, j = stack[-1]
        options = []
        for di, dj in _STEPS:
            ni, nj = i + di, j + dj
            if (0 <= ni < rows and 0 <= nj < cols
                    and not visited[ni * cols + nj]):
                options.append((ni, nj, di, dj))
        if not options:
            stack.pop()
            continue
        ni, nj, di, dj = options[rng.randrange(len(options))]
        visited[ni * cols + nj] = 1
        grid[2 * i + 1 + di][2 * j + 1 + dj] = _SPACE
        grid[2 * ni + 1][2 * nj + 1] = _SPACE
        stack.append((ni, nj))


def _prim(grid: Grid, rows: int, cols: int, rng: random.Random) -> None:
    """Carve a perfect maze with randomised Prim's algorithm."""
    visited = bytearray(rows * cols)
    queued = bytearray(rows * cols)
    frontier = []

    def visit(i: int, j: int) -> None:
        visited[i * cols + j] = 1
        grid[2 * i + 1][2 * j + 1] = _SPACE
        for di, dj in _STEPS:
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                index = ni * cols + nj
                if not visited[index] and not queued[index]:
                    queued[index] = 1
                    frontier.append((ni, nj))

    visit(rng.randrange(rows), rng.randrange(cols))
    while frontier:
        pick = rng.randrange(len(frontier))
        frontier[pick], frontier[-1] = frontier[-1], frontier[pick]
        i, j = frontier.pop()
        links = [(di, dj) for di, dj in _STEPS
                 if 0 <= i + di < rows and 0 <= j + dj < cols
                 and visited[(i + di) * cols + j + dj]]
        di, dj = links[rng.randrange(len(links))]
        grid[2 * i + 1 + di][2 * j + 1 + dj] = _SPACE
        visit(i, j)


def _binary_tree(grid: Grid, rows: int, cols: int,
                 rng: random.Random) -> None:
    """Carve a perfect maze in which every cell opens north or east."""
    for i in range(rows):
        for j in range(cols):
            grid[2 * i + 1][2 * j + 1] = _SPACE
            options = []
            if i > 0:
                options.append((-1, 0))
            if j < cols - 1:
                options.append((0, 1))
            if options:
                di, dj = options[rng.randrange(len(options))]
                grid[2 * i + 1 + di][2 * j + 1 + dj] = _SPACE


_CARVERS: Dict[str, Callable[[Grid, int, int, random.Random], None]] = {
    "backtracker": _backtracker,
    "prim": _prim,
    "binary_tree": _binary_tree,
}


def generate_maze(width: int = 41, height: Optional[int] = None,
                  algorithm: str = "backtracker",
                  seed: Optional[int] = None) -> Maze:
    """Generate a perfect maze with S top-left and E bottom-right.

    Args:
        width: Grid width in characters (odd sizes give a full border).
        height: Grid height; defaults to ``width``.
        algorithm: One of :data:`GENERATORS`.
        seed: Seed for the random number generator (``None`` = random).

    Returns:
        The generated maze.

    Raises:
        ValueError: For an unknown algorithm or a grid too small to hold
            two corridor cells.
    """
    height = width if height is None else height
    if algorithm not in _CARVERS:
        raise ValueError(f"unknown generator {algorithm!r}; "
                         f"choose from {', '.join(GENERATORS)}")
    rows, cols = (height - 1) // 2, (width - 1) // 2
    if rows < 1 or cols < 1 or rows * cols < 2:
        raise ValueError("the grid is too small; use at least 5 x 3")
    grid = [bytearray(WALL.encode() * width) for _ in range(height)]
    _CARVERS[algorithm](grid, rows, cols, random.Random(seed))
    start, goal = (1, 1), (2 * rows - 1, 2 * cols - 1)
    grid[start[0]][start[1]] = ord(START)
    grid[goal[0]][goal[1]] = ord(GOAL)
    return Maze(tuple(row.decode("ascii") for row in grid), start, goal)


def removable_walls(maze: Maze) -> List[Cell]:
    """Interior wall cells that separate two corridor cells in line.

    In a perfect maze, opening any one of them adds exactly one loop.
    """
    rows = maze.rows
    walls = []
    for r in range(1, maze.height - 1):
        for c in range(1, maze.width - 1):
            if rows[r][c] != WALL:
                continue
            left_right = (rows[r][c - 1] != WALL and rows[r][c + 1] != WALL
                          and rows[r - 1][c] == WALL
                          and rows[r + 1][c] == WALL)
            up_down = (rows[r - 1][c] != WALL and rows[r + 1][c] != WALL
                       and rows[r][c - 1] == WALL and rows[r][c + 1] == WALL)
            if left_right or up_down:
                walls.append((r, c))
    return walls


def _opened(maze: Maze, cells: Sequence[Cell]) -> Maze:
    """Return a copy of ``maze`` with the given cells turned into floor."""
    grid = [list(row) for row in maze.rows]
    for r, c in cells:
        grid[r][c] = " "
    return Maze(tuple("".join(row) for row in grid), maze.start, maze.goal)


def braid(maze: Maze, fraction: float, seed: Optional[int] = None) -> Maze:
    """Open a random ``fraction`` of the removable walls to add loops.

    Args:
        maze: The maze to modify (usually a perfect maze).
        fraction: Share of removable walls to open, between 0 and 1.
        seed: Seed for choosing the walls.

    Returns:
        A new maze with the walls opened.
    """
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    candidates = removable_walls(maze)
    count = round(fraction * len(candidates))
    return _opened(maze, random.Random(seed).sample(candidates, count))


def without_interior_walls(maze: Maze) -> Maze:
    """Remove every wall inside the bounding box of the open cells."""
    rows = [r for r, row in enumerate(maze.rows) if row.strip(WALL)]
    cols = [c for c in range(maze.width)
            if any(row[c] != WALL for row in maze.rows)]
    cells = [(r, c) for r in range(rows[0], rows[-1] + 1)
             for c in range(cols[0], cols[-1] + 1)
             if maze.rows[r][c] == WALL]
    return _opened(maze, cells)


def with_endpoints(maze: Maze, start: Optional[Cell] = None,
                   goal: Optional[Cell] = None) -> Maze:
    """Return a copy of ``maze`` with S and/or E moved to open cells."""
    start = maze.start if start is None else start
    goal = maze.goal if goal is None else goal
    if start == goal:
        raise ValueError("the start and the exit must differ")
    for cell in (start, goal):
        if not maze.is_open(cell):
            raise ValueError(f"{cell} is not an open cell")
    grid = [list(row) for row in maze.rows]
    grid[maze.start[0]][maze.start[1]] = " "
    grid[maze.goal[0]][maze.goal[1]] = " "
    grid[start[0]][start[1]] = START
    grid[goal[0]][goal[1]] = GOAL
    return Maze(tuple("".join(row) for row in grid), start, goal)


def pick_cell(maze: Maze, where: str = "corner",
              seed: Optional[int] = None) -> Cell:
    """Pick an open cell: ``corner`` (bottom-right), ``center`` or ``random``.

    The start cell is never picked.
    """
    cells = [(r, c) for r, row in enumerate(maze.rows)
             for c, char in enumerate(row)
             if char != WALL and (r, c) != maze.start]
    if where == "corner":
        return max(cells, key=lambda cell: (cell[0] + cell[1], cell))
    if where == "center":
        mid_r, mid_c = (maze.height - 1) / 2, (maze.width - 1) / 2
        return min(cells, key=lambda cell: (abs(cell[0] - mid_r)
                                            + abs(cell[1] - mid_c), cell))
    if where == "random":
        return random.Random(seed).choice(cells)
    raise ValueError("where must be 'corner', 'center' or 'random'")


def save_maze(maze: Maze, path: str) -> None:
    """Write ``maze`` to ``path``; a ``.gz`` suffix gzip-compresses it.

    Compressed output is byte-identical for the same maze, whatever
    the file is called, which keeps Git diffs clean.
    """
    data = maze_to_text(maze).encode("ascii")
    if str(path).endswith(".gz"):
        # An empty stored file name and a fixed timestamp keep the
        # compressed bytes identical however the file is named.
        with open(path, "wb") as raw, gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, compresslevel=9,
                mtime=0) as handle:
            handle.write(data)
    else:
        with open(path, "wb") as handle:
            handle.write(data)
