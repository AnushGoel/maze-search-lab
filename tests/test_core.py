"""Tests for parsing, the three classic searches and the command line."""

import gzip
import os
import tempfile

import pytest

from mazesearch import (a_star_search, breadth_first_search,
                        depth_first_search, load_maze, maze_solver_one,
                        maze_solver_three, maze_solver_two, maze_to_text)
from mazesearch.cli import main

SOLVERS = (maze_solver_one, maze_solver_two, maze_solver_three)
CORRIDOR = "5 3\nXXXXX\nXS EX\nXXXXX\n"


def test_corridor_is_solved_exactly():
    for solver in SOLVERS:
        assert solver(CORRIDOR) == "5 3\nXXXXX\nXS*EX\nXXXXX\n"


def test_unsolvable_maze_returns_none():
    for solver in SOLVERS:
        assert solver("6 3\nXXXXXX\nXSX EX\nXXXXXX\n") is None


def test_answer_comes_back_in_the_input_form():
    lines = CORRIDOR.splitlines(keepends=True)
    assert maze_solver_two(lines)[2] == "XS*EX\n"
    grid = [list(row) for row in CORRIDOR.splitlines()[1:]]
    assert maze_solver_three(grid)[1] == list("XS*EX")
    assert maze_solver_one("XXXXX\nXS EX\nXXXXX") == "XXXXX\nXS*EX\nXXXXX"


def test_plain_and_gzip_files_give_the_same_answer():
    with tempfile.TemporaryDirectory() as folder:
        plain = os.path.join(folder, "maze.txt")
        packed = os.path.join(folder, "maze.txt.gz")
        with open(plain, "w", encoding="utf-8") as handle:
            handle.write(CORRIDOR)
        with gzip.open(packed, "wt", encoding="utf-8") as handle:
            handle.write(CORRIDOR)
        assert maze_solver_one(plain) == maze_solver_one(packed)


def test_invalid_mazes_raise_value_error():
    for bad in ("3 1\nX E\n", "4 1\nSSE \n", "4 1\nSEE \n"):
        with pytest.raises(ValueError):
            load_maze(bad)


def test_text_round_trip():
    maze = load_maze(CORRIDOR)
    assert load_maze(maze_to_text(maze)) == maze


def test_counters_are_consistent():
    maze = load_maze("7 5\nXXXXXXX\nXS    X\nX XXX X\nX    EX\nXXXXXXX\n")
    for search in (breadth_first_search, depth_first_search, a_star_search):
        result = search(maze)
        assert result.solved and result.path_length == 6
        assert result.nodes_generated >= result.nodes_expanded
        assert result.path[0] == maze.start and result.path[-1] == maze.goal


def test_command_line_round_trip():
    with tempfile.TemporaryDirectory() as folder:
        target = os.path.join(folder, "maze.txt.gz")
        assert main(["generate", "--width", "21", "--seed", "1",
                     "-o", target]) == 0
        assert main(["solve", target, "-a", "all"]) == 0
        assert main(["stats", target]) == 0
