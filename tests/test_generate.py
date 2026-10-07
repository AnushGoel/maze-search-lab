"""Tests for the maze generators and maze transformations."""

import os
import tempfile

import pytest

from mazesearch import GENERATORS, braid, generate_maze, load_maze
from mazesearch.analysis import maze_stats
from mazesearch.generate import (pick_cell, removable_walls, save_maze,
                                 with_endpoints, without_interior_walls)


def test_generators_make_perfect_mazes():
    for generator in GENERATORS:
        for size in (5, 21, 40):
            maze = generate_maze(size, size + 4, generator, seed=size)
            stats = maze_stats(maze)
            assert stats["loops"] == 0, (generator, size)
            assert stats["regions"] == 1, (generator, size)


def test_generation_is_deterministic():
    for generator in GENERATORS:
        first = generate_maze(31, algorithm=generator, seed=9)
        assert first == generate_maze(31, algorithm=generator, seed=9)


def test_braiding_adds_loops():
    maze = generate_maze(41, seed=1)
    assert braid(maze, 0.0, 1) == maze
    opened = round(0.25 * len(removable_walls(maze)))
    assert maze_stats(braid(maze, 0.25, 1))["loops"] >= opened > 0


def test_open_room_and_moved_exit():
    room = without_interior_walls(generate_maze(21, seed=2))
    assert maze_stats(room)["tortuosity"] == 1.0
    goal = pick_cell(room, "center")
    moved = with_endpoints(room, goal=goal)
    assert moved.goal == goal and moved.rows[goal[0]][goal[1]] == "E"
    with pytest.raises(ValueError):
        with_endpoints(room, goal=(0, 0))


def test_saved_files_are_byte_identical():
    maze = generate_maze(25, seed=4)
    with tempfile.TemporaryDirectory() as folder:
        first = os.path.join(folder, "a.txt.gz")
        second = os.path.join(folder, "b.txt.gz")
        save_maze(maze, first)
        save_maze(maze, second)
        with open(first, "rb") as one, open(second, "rb") as two:
            assert one.read() == two.read()
        assert load_maze(first) == maze


def test_bad_arguments_raise():
    with pytest.raises(ValueError):
        generate_maze(3)
    with pytest.raises(ValueError):
        generate_maze(21, algorithm="not-a-generator")
