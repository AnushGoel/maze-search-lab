"""Command-line interface: ``python -m mazesearch <command>``.

Examples::

    python -m mazesearch generate --width 61 --generator prim -o maze.txt
    python -m mazesearch solve maze.txt --algorithms bfs dfs astar alt
    python -m mazesearch stats maze.txt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional, Sequence

from . import __version__
from .algorithms import ALGORITHMS
from .analysis import heuristic_profile, maze_stats
from .core import load_maze, maze_to_text, render_solution
from .generate import GENERATORS, braid, generate_maze, save_maze


def _solve(args: argparse.Namespace) -> int:
    """Solve a maze file with the selected algorithms."""
    maze = load_maze(args.maze)
    keys = list(ALGORITHMS) if "all" in args.algorithms else args.algorithms
    print(f"{args.maze}: {maze.width} x {maze.height}, "
          f"start {maze.start}, exit {maze.goal}")
    print(f"{'algorithm':<16}{'path':>10}{'expanded':>12}"
          f"{'frontier':>10}{'seconds':>10}")
    solved = False
    for key in keys:
        started = time.perf_counter()
        result = ALGORITHMS[key].search(maze)
        elapsed = time.perf_counter() - started
        length = "-" if result.path is None else f"{result.path_length:,}"
        print(f"{result.algorithm:<16}{length:>10}"
              f"{result.nodes_expanded:>12,}{result.max_frontier:>10,}"
              f"{elapsed:>10.3f}")
        if result.path is None:
            continue
        solved = True
        if args.show:
            print(render_solution(maze, result.path))
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            stem = os.path.basename(args.maze).split(".")[0]
            target = os.path.join(args.output_dir, f"{stem}_{key}.txt")
            with open(target, "w", encoding="utf-8") as handle:
                handle.write(render_solution(maze, result.path))
    return 0 if solved else 1


def _generate(args: argparse.Namespace) -> int:
    """Generate a maze and print or save it."""
    maze = generate_maze(args.width, args.height, args.generator, args.seed)
    if args.loops:
        maze = braid(maze, args.loops, args.seed)
    if args.output:
        save_maze(maze, args.output)
        print(f"saved {maze.width} x {maze.height} maze to {args.output}")
    else:
        sys.stdout.write(maze_to_text(maze))
    return 0


def _stats(args: argparse.Namespace) -> int:
    """Print structural statistics and the A* heuristic profile."""
    maze = load_maze(args.maze)
    stats = maze_stats(maze)
    for name, value in stats.items():
        shown = f"{value:.2f}" if isinstance(value, float) else value
        print(f"{name:<18}{shown}")
    if stats["shortest_path"] is not None:
        for name, value in heuristic_profile(maze).items():
            print(f"{name:<18}{value:,}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for all sub-commands."""
    parser = argparse.ArgumentParser(
        prog="mazesearch",
        description="Blind and heuristic search on grid mazes.")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    solve = commands.add_parser("solve", help="solve a maze file")
    solve.add_argument("maze", help="maze file (.txt or .txt.gz)")
    solve.add_argument("-a", "--algorithms", nargs="+",
                       default=["bfs", "dfs", "astar"],
                       choices=[*ALGORITHMS, "all"])
    solve.add_argument("--show", action="store_true",
                       help="print each solved maze")
    solve.add_argument("-o", "--output-dir",
                       help="save solved mazes in this folder")
    solve.set_defaults(handler=_solve)

    generate = commands.add_parser("generate", help="generate a maze")
    generate.add_argument("--width", type=int, default=41)
    generate.add_argument("--height", type=int, default=None)
    generate.add_argument("--generator", choices=GENERATORS,
                          default="backtracker")
    generate.add_argument("--loops", type=float, default=0.0,
                          help="share of removable walls to open (0-1)")
    generate.add_argument("--seed", type=int, default=None)
    generate.add_argument("-o", "--output",
                          help="output file (.gz to compress)")
    generate.set_defaults(handler=_generate)

    stats = commands.add_parser("stats", help="describe a maze file")
    stats.add_argument("maze")
    stats.set_defaults(handler=_stats)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command line; returns the process exit status."""
    args = build_parser().parse_args(argv)
    return args.handler(args)
