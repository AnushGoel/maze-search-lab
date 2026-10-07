"""Tests for heuristics and the analysis tools behind the findings."""

from mazesearch import (LandmarkHeuristic, a_star_search, braid,
                        breadth_first_search, depth_first_search,
                        distance_table, euclidean_distance, generate_maze,
                        manhattan_distance)
from mazesearch.analysis import exit_census, heuristic_profile
from mazesearch.generate import pick_cell, with_endpoints


def test_distance_table_matches_bfs():
    maze = braid(generate_maze(31, algorithm="prim", seed=5), 0.3, 5)
    table = distance_table(maze, maze.start)
    for seed in range(10):
        goal = pick_cell(maze, "random", seed)
        route = breadth_first_search(with_endpoints(maze, goal=goal))
        assert table[goal[0] * maze.width + goal[1]] == route.path_length


def test_heuristics_are_admissible_and_consistent():
    maze = braid(generate_maze(41, seed=8), 0.2, 8)
    landmarks = LandmarkHeuristic.farthest(maze, 6)
    truth = distance_table(maze, maze.goal)
    for index, true_cost in enumerate(truth):
        if true_cost < 0:
            continue
        cell = divmod(index, maze.width)
        bound = landmarks(cell, maze.goal)
        manhattan = manhattan_distance(cell, maze.goal)
        assert bound <= true_cost and manhattan <= true_cost
        assert euclidean_distance(cell, maze.goal) <= manhattan + 1e-9
        for neighbor in maze.neighbors(cell):
            assert abs(bound - landmarks(neighbor, maze.goal)) <= 1


def test_exit_census_matches_real_searches():
    maze = braid(generate_maze(35, seed=3), 0.1, 3)
    bfs_cost, dfs_cost = exit_census(maze)
    for seed in range(25):
        goal = pick_cell(maze, "random", seed)
        variant = with_endpoints(maze, goal=goal)
        index = goal[0] * maze.width + goal[1]
        assert breadth_first_search(variant).nodes_expanded == bfs_cost[index]
        assert depth_first_search(variant).nodes_expanded == dfs_cost[index]


def test_a_star_stays_inside_the_theoretical_window():
    for seed in range(5):
        maze = generate_maze(51, algorithm="prim", seed=seed)
        profile = heuristic_profile(maze)
        assert profile["must_expand"] <= profile["may_expand"]
        assert profile["prunable"] == (profile["closer_than_exit"]
                                       - profile["must_expand"])
        expanded = a_star_search(maze).nodes_expanded
        assert profile["must_expand"] <= expanded <= profile["may_expand"]
