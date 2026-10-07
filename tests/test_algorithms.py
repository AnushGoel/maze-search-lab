"""Every registered algorithm returns a valid route; optimal ones are
always as short as BFS's route."""

from mazesearch import (ALGORITHMS, braid, breadth_first_search,
                        generate_maze, load_maze)
from mazesearch.generate import pick_cell, with_endpoints


def is_valid_route(maze, path):
    """True if ``path`` walks from S to E through open, adjacent cells."""
    if path[0] != maze.start or path[-1] != maze.goal:
        return False
    if len(set(path)) != len(path):
        return False
    return all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 and maze.is_open(b)
               for a, b in zip(path, path[1:]))


def random_mazes(count=24):
    """Seeded mazes of every texture, with and without loops."""
    for seed in range(count):
        generator = ("backtracker", "prim", "binary_tree")[seed % 3]
        maze = generate_maze(21 + 2 * (seed % 6), algorithm=generator,
                             seed=seed)
        maze = braid(maze, (0.0, 0.2, 0.5, 1.0)[seed % 4], seed)
        yield with_endpoints(maze, goal=pick_cell(maze, "random", seed))


def test_all_algorithms_on_random_mazes():
    for maze in random_mazes():
        shortest = breadth_first_search(maze).path_length
        for key, algorithm in ALGORITHMS.items():
            result = algorithm.search(maze)
            assert is_valid_route(maze, result.path), key
            if algorithm.optimal:
                assert result.path_length == shortest, key
            else:
                assert result.path_length >= shortest, key


def test_every_algorithm_reports_an_unsolvable_maze():
    maze = load_maze("7 3\nXXXXXXX\nXS X EX\nXXXXXXX\n")
    for key, algorithm in ALGORITHMS.items():
        assert algorithm.search(maze).path is None, key


def test_result_labels_match_the_registry():
    maze = generate_maze(15, seed=3)
    for key, algorithm in ALGORITHMS.items():
        assert algorithm.search(maze).algorithm == algorithm.label, key
