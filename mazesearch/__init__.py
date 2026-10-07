"""mazesearch: blind and heuristic search on grid mazes.

Quick start::

    from mazesearch import generate_maze, ALGORITHMS
    maze = generate_maze(101, seed=7)
    for key in ("bfs", "dfs", "astar"):
        result = ALGORITHMS[key].search(maze)
        print(result.algorithm, result.path_length, result.nodes_expanded)
"""

__version__ = "1.0.0"

from .algorithms import (ALGORITHMS, Algorithm, bidirectional_bfs,  # noqa
                         greedy_best_first_search, landmark_a_star_search,
                         uniform_cost_search, weighted_a_star_search)
from .core import (Maze, SearchResult, a_star_search,  # noqa: E402
                   breadth_first_search, depth_first_search, load_maze,
                   manhattan_distance, mark_path, maze_solver_one,
                   maze_solver_three, maze_solver_two, maze_to_text,
                   render_solution, solve_maze)
from .generate import GENERATORS, braid, generate_maze  # noqa: E402
from .heuristics import (LandmarkHeuristic, distance_table,  # noqa: E402
                         euclidean_distance, weighted, zero_heuristic)

__all__ = [
    "ALGORITHMS", "Algorithm", "GENERATORS", "LandmarkHeuristic", "Maze",
    "SearchResult", "a_star_search", "bidirectional_bfs", "braid",
    "breadth_first_search", "depth_first_search", "distance_table",
    "euclidean_distance", "generate_maze", "greedy_best_first_search",
    "landmark_a_star_search", "load_maze", "manhattan_distance",
    "mark_path", "maze_solver_one", "maze_solver_three", "maze_solver_two",
    "maze_to_text", "render_solution", "solve_maze", "uniform_cost_search",
    "weighted", "weighted_a_star_search", "zero_heuristic",
]
