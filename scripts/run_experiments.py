"""Run every experiment and write the datasets used by the app and docs.

Usage::

    python scripts/run_experiments.py                 # full study, ~8 min
    python scripts/run_experiments.py --quick         # smoke run, ~30 s
    python scripts/run_experiments.py --maze my.txt   # study your own maze

Every random choice is seeded, and every count (cells expanded, path
lengths, frontier sizes) is deterministic, so a re-run reproduces the
published numbers exactly; only the timings depend on the machine.
Results go to ``data/results``: small CSV/JSON tables plus one
compressed NPZ file of downsampled maps for the app.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mazesearch import (ALGORITHMS, GENERATORS, LandmarkHeuristic,  # noqa
                        __version__, a_star_search, bidirectional_bfs, braid,
                        breadth_first_search, depth_first_search,
                        distance_table, euclidean_distance, generate_maze,
                        greedy_best_first_search, landmark_a_star_search,
                        load_maze, manhattan_distance, uniform_cost_search,
                        weighted, weighted_a_star_search, zero_heuristic)
from mazesearch.analysis import (exit_census, expansion_order,  # noqa
                                 heuristic_profile, maze_stats)
from mazesearch.bench import (interleaved_times, peak_memory_mib,  # noqa
                              summarize)
from mazesearch.core import MOVES  # noqa: E402
from mazesearch.generate import (save_maze,  # noqa: E402
                                 with_endpoints, without_interior_walls)
from mazesearch.render import downsample, wall_mask  # noqa: E402

SEED = 2026
OUT = ROOT / "data" / "results"
COMPASS = {(-1, 0): "N", (0, 1): "E", (1, 0): "S", (0, -1): "W"}
TIMINGS = {}


def log(message: str) -> None:
    """Print a timestamped progress line."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def timed(name: str):
    """Decorator that records how long an experiment took."""
    def wrap(func):
        @functools.wraps(func)
        def run(*args, **kwargs):
            started = time.perf_counter()
            log(f"{name} ...")
            value = func(*args, **kwargs)
            TIMINGS[name] = round(time.perf_counter() - started, 1)
            log(f"{name} done in {TIMINGS[name]} s")
            return value
        return run
    return wrap


def cpu_name() -> str:
    """CPU model name where the operating system exposes it."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def save(frame: pd.DataFrame, name: str) -> None:
    """Write a results table as CSV with compact number formatting."""
    frame.to_csv(OUT / f"{name}.csv", index=False, float_format="%.6g")


def to_uint8(values: np.ndarray) -> np.ndarray:
    """Quantise values in [0, 1] to 0-254, with 255 marking NaN."""
    clean = np.nan_to_num(values, nan=0.0)
    out = np.round(np.clip(clean, 0, 1) * 254).astype(np.uint8)
    out[np.isnan(values)] = 255
    return out


def grid_of(flat, maze) -> np.ndarray:
    """Flat per-cell array -> (height, width) float grid, NaN for -1."""
    grid = np.frombuffer(flat, dtype=np.int32).astype(float)
    grid = grid.reshape(maze.height, maze.width)
    grid[grid < 0] = np.nan
    return grid


def update_json(name: str, **fields) -> None:
    """Merge ``fields`` into a JSON file, so stages can run separately."""
    path = OUT / name
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(fields)
    path.write_text(json.dumps(data, indent=1))


def update_maps(new_maps: dict) -> None:
    """Merge arrays into ``maps.npz`` (compressed)."""
    path = OUT / "maps.npz"
    merged = {}
    if path.exists():
        with np.load(path) as data:
            merged = {key: data[key] for key in data.files}
    merged.update(new_maps)
    np.savez_compressed(path, **merged)


def spearman(x, y) -> float:
    """Spearman rank correlation (Spearman, 1904), ties averaged."""
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


@timed("E1 benchmark")
def benchmark(maze, alt8, repeats):
    """Time, memory and work of eight algorithms on the study maze."""
    searches = {
        "BFS": breadth_first_search,
        "DFS": depth_first_search,
        "A*": a_star_search,
        "Bi-BFS": bidirectional_bfs,
        "UCS": uniform_cost_search,
        "WA* (w=2)": weighted_a_star_search,
        "Greedy": greedy_best_first_search,
        "A* + ALT (k=8)": functools.partial(landmark_a_star_search,
                                            heuristic=alt8),
    }
    results = {name: search(maze) for name, search in searches.items()}
    times = interleaved_times(maze, searches, repeats=repeats, warmup=1)
    shortest = results["BFS"].path_length
    open_count = sum(len(row) - row.count("X") for row in maze.rows)
    family = {"BFS": "blind", "DFS": "blind", "Bi-BFS": "blind",
              "UCS": "blind"}
    rows, runs = [], []
    for name, result in results.items():
        stats = summarize(times[name])
        rows.append({
            "algorithm": name, "family": family.get(name, "informed"),
            "path_length": result.path_length,
            "optimal": result.path_length == shortest,
            "nodes_expanded": result.nodes_expanded,
            "nodes_generated": result.nodes_generated,
            "peak_frontier": result.max_frontier,
            "nodes_reached": result.nodes_reached,
            "pct_open_expanded": 100 * result.nodes_expanded / open_count,
            "peak_memory_mib": peak_memory_mib(searches[name], maze),
            "median_s": stats["median"], "q1_s": stats["q1"],
            "q3_s": stats["q3"], "iqr_s": stats["iqr"],
            "us_per_expansion": 1e6 * stats["median"]
            / max(result.nodes_expanded, 1),
        })
        runs += [{"algorithm": name, "run": k + 1, "seconds": t}
                 for k, t in enumerate(times[name])]
    save(pd.DataFrame(rows), "benchmark")
    save(pd.DataFrame(runs), "benchmark_runs")
    update_json("summary.json", astar_saving_vs_bfs=(
        results["BFS"].nodes_expanded - results["A*"].nodes_expanded))


@timed("E2 heuristic analysis")
def heuristic_analysis(maze, alt16, factor):
    """Why the Manhattan heuristic barely helps, plus the app's maps."""
    results = {"BFS": breadth_first_search(maze)}
    goal, width = maze.goal, maze.width
    g_flat = distance_table(maze, maze.start)
    h_flat = distance_table(maze, goal)
    g = np.frombuffer(g_flat, dtype=np.int32)
    h_true = np.frombuffer(h_flat, dtype=np.int32)
    reach = g >= 0
    rows_idx, cols_idx = np.divmod(np.arange(g.size), width)
    manhattan = np.abs(rows_idx - goal[0]) + np.abs(cols_idx - goal[1])
    c_star = int(g[goal[0] * width + goal[1]])

    profiles = {"Manhattan": heuristic_profile(maze)}
    for k in (4, 8, 16):
        profiles[f"ALT k={k}"] = heuristic_profile(maze, alt16.subset(k))
    profiles["Exact h*"] = {"must_expand": 0}

    # Distribution of true distances from S (for the pruning band).
    counts, edges = np.histogram(g[reach], bins=150)
    save(pd.DataFrame({"bin_start": edges[:-1], "bin_end": edges[1:],
                       "cells": counts}), "depth_hist")

    # Heuristic versus truth along the route.
    route = results["BFS"].path
    alt8 = alt16.subset(8)
    picks = np.unique(np.linspace(0, len(route) - 1, 400).astype(int))
    save(pd.DataFrame([{
        "position": int(p), "h_true": int(h_true[route[p][0] * width
                                                 + route[p][1]]),
        "manhattan": manhattan_distance(route[p], goal),
        "alt8": alt8(route[p], goal)} for p in picks]), "route_profile")

    # How good must a heuristic be? Blend Manhattan towards the truth.
    gr, mr, hr = g[reach], manhattan[reach], h_true[reach]
    curve = [{"quality": q, "must_expand": int(np.sum(
        gr + (1 - q) * mr + q * hr < c_star))}
        for q in np.round(np.linspace(0, 1, 21), 2)]
    save(pd.DataFrame(curve), "quality_curve")

    # Heuristic ablation (same A* code, different h).
    variants = {
        "h = 0 (uniform-cost)": zero_heuristic,
        "Euclidean": euclidean_distance,
        "Manhattan": manhattan_distance,
        "Weighted, w = 2": weighted(manhattan_distance, 2),
        "Weighted, w = 5": weighted(manhattan_distance, 5),
        "Greedy-like, w = 10^6": weighted(manhattan_distance, 10 ** 6),
        "ALT k=4": alt16.subset(4), "ALT k=8": alt16.subset(8),
        "ALT k=16": alt16,
    }
    rows = []
    for label, heuristic in variants.items():
        seconds = []
        for _ in range(3):
            started = time.perf_counter()
            result = a_star_search(maze, heuristic)
            seconds.append(time.perf_counter() - started)
        consistent = not label.startswith(("Weighted", "Greedy"))
        must = (heuristic_profile(maze, heuristic)["must_expand"]
                if consistent and label not in ("Manhattan",)
                and not label.startswith("ALT") else None)
        if label == "Manhattan":
            must = profiles["Manhattan"]["must_expand"]
        if label.startswith("ALT"):
            must = profiles[label]["must_expand"]
        rows.append({"heuristic": label, "consistent": consistent,
                     "path_length": result.path_length,
                     "nodes_expanded": result.nodes_expanded,
                     "pct_of_bfs": 100 * result.nodes_expanded
                     / results["BFS"].nodes_expanded,
                     "must_expand": must,
                     "median_s": float(np.median(seconds))})
    save(pd.DataFrame(rows), "heuristics")

    # Compact maps for the app (downsampled, quantised to uint8).
    maps = {}
    for key, search in (("BFS", breadth_first_search),
                        ("DFS", depth_first_search),
                        ("A*", a_star_search)):
        order, _ = expansion_order(maze, search)
        grid = grid_of(order, maze)
        maps[f"order_{key}"] = to_uint8(
            downsample(grid / np.nanmax(grid), factor))
    order, _ = expansion_order(maze, a_star_search, heuristic=alt8)
    grid = grid_of(order, maze)
    maps["order_ALT8"] = to_uint8(downsample(grid / np.nanmax(grid), factor))
    dist = grid_of(g_flat, maze)
    maps["distance"] = to_uint8(downsample(dist / np.nanmax(dist), factor))
    maps["walls"] = to_uint8(downsample(wall_mask(maze).astype(float),
                                        factor))
    route_grid = np.zeros((maze.height, maze.width))
    route_grid[tuple(np.array(route).T)] = 1
    maps["route"] = (downsample(route_grid, factor, np.nanmax) > 0
                     ).astype(np.uint8)
    update_maps(maps)
    update_json("summary.json", profiles=profiles, c_star=c_star)


@timed("E3 exit census")
def census(maze, factor, samples):
    """Cost of every possible exit for BFS and DFS, plus a check."""
    bfs_cost, dfs_cost = exit_census(maze)
    bfs = np.frombuffer(bfs_cost, dtype=np.int32).astype(float)
    dfs = np.frombuffer(dfs_cost, dtype=np.int32).astype(float)
    dist = np.frombuffer(distance_table(maze, maze.start),
                         dtype=np.int32).astype(float)
    valid = (bfs >= 0) & (dfs >= 0)
    ratio = np.full(bfs.size, np.nan)
    ratio[valid] = np.log2((dfs[valid] + 1) / (bfs[valid] + 1))
    grid = ratio.reshape(maze.height, maze.width)
    luck = downsample(grid, factor, np.nanmedian)
    update_maps({"luck": np.where(np.isnan(luck), -128, np.clip(
        np.round(luck * 16), -127, 127)).astype(np.int8)})

    edges = np.unique(np.geomspace(1, dist[valid].max() + 1, 41).astype(int))
    bins = []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = valid & (dist >= low) & (dist < high)
        if inside.any():
            bins.append({"distance_low": int(low), "distance_high": int(high),
                         "exits": int(inside.sum()),
                         "bfs_median": float(np.median(bfs[inside])),
                         "dfs_median": float(np.median(dfs[inside])),
                         "dfs_q1": float(np.percentile(dfs[inside], 25)),
                         "dfs_q3": float(np.percentile(dfs[inside], 75)),
                         "dfs_wins_pct": float(100 * np.mean(
                             dfs[inside] < bfs[inside]))})
    save(pd.DataFrame(bins), "exit_bins")

    # Verify the census against real searches on random exits.
    rng = random.Random(SEED)
    cells = [divmod(int(i), maze.width) for i in np.flatnonzero(valid)]
    chosen = rng.sample(cells, samples) + [maze.goal]
    rows = []
    for goal in chosen:
        variant = with_endpoints(maze, goal=goal)
        index = goal[0] * maze.width + goal[1]
        real = {name: search(variant).nodes_expanded for name, search in
                (("bfs", breadth_first_search), ("dfs", depth_first_search),
                 ("astar", a_star_search))}
        rows.append({"row": goal[0], "col": goal[1],
                     "distance": int(dist[index]),
                     "manhattan": manhattan_distance(maze.start, goal),
                     "bfs": real["bfs"], "dfs": real["dfs"],
                     "astar": real["astar"],
                     "census_bfs": int(bfs[index]),
                     "census_dfs": int(dfs[index]),
                     "original_exit": goal == maze.goal})
    sample = pd.DataFrame(rows)
    save(sample, "exits_sample")
    matches = int(((sample.bfs == sample.census_bfs)
                   & (sample.dfs == sample.census_dfs)).sum())
    update_json("summary.json", census={"exits": int(valid.sum()),
            "dfs_cheaper_pct": float(100 * np.mean(dfs[valid] < bfs[valid])),
            "median_log2_ratio": float(np.median(ratio[valid])),
            "spearman_bfs": spearman(dist[valid], bfs[valid]),
            "spearman_dfs": spearman(dist[valid], dfs[valid]),
            "astar_over_bfs_median": float(np.median(sample.astar
                                                     / sample.bfs)),
            "census_checks": len(sample), "census_matches": matches})


@timed("E4 loops")
def loops(maze, fractions, seeds):
    """Open a growing share of walls and rerun the searches."""
    searches = {"BFS": breadth_first_search, "DFS": depth_first_search,
                "A*": a_star_search, "Bi-BFS": bidirectional_bfs}
    rows = []
    for fraction in fractions:
        for seed in (seeds if fraction else [0]):
            braided = braid(maze, fraction, SEED + seed)
            open_count = sum(len(r) - r.count("X") for r in braided.rows)
            shortest = None
            for name, search in searches.items():
                started = time.perf_counter()
                result = search(braided)
                elapsed = time.perf_counter() - started
                shortest = shortest or result.path_length
                rows.append({"fraction": fraction, "seed": seed,
                             "algorithm": name,
                             "path_length": result.path_length,
                             "shortest": shortest,
                             "path_ratio": result.path_length / shortest,
                             "nodes_expanded": result.nodes_expanded,
                             "pct_expanded": 100 * result.nodes_expanded
                             / open_count,
                             "open_cells": open_count, "seconds": elapsed})
    save(pd.DataFrame(rows), "loops")


@timed("E5 move order")
def move_orders(maze):
    """DFS effort for all 24 orders of the four moves."""
    rows = []
    for moves in itertools.permutations(MOVES):
        label = "-".join(COMPASS[m] for m in moves)
        result = depth_first_search(maze, moves)
        rows.append({"order": label, "nodes_expanded": result.nodes_expanded,
                     "peak_frontier": result.max_frontier,
                     "south_before_north": label.index("S")
                     < label.index("N"),
                     "submitted": moves == MOVES})
    save(pd.DataFrame(rows).sort_values("nodes_expanded"), "move_orders")


@timed("E6 open room")
def open_room(maze):
    """Remove every interior wall: the best case for the heuristic."""
    room = without_interior_walls(maze)
    nwse = ((-1, 0), (0, -1), (1, 0), (0, 1))
    searches = {
        "BFS": breadth_first_search,
        "DFS (N-E-S-W)": depth_first_search,
        "DFS (N-W-S-E)": functools.partial(depth_first_search, moves=nwse),
        "A*": a_star_search, "Bi-BFS": bidirectional_bfs,
    }
    open_count = sum(len(r) - r.count("X") for r in room.rows)
    shortest = manhattan_distance(room.start, room.goal)
    rows = []
    for name, search in searches.items():
        started = time.perf_counter()
        result = search(room)
        rows.append({"algorithm": name, "path_length": result.path_length,
                     "path_ratio": result.path_length / shortest,
                     "nodes_expanded": result.nodes_expanded,
                     "pct_expanded": 100 * result.nodes_expanded
                     / open_count,
                     "peak_frontier": result.max_frontier,
                     "seconds": time.perf_counter() - started})
    save(pd.DataFrame(rows), "open_room")


@timed("E7 texture and scaling")
def texture_and_scaling(sizes, seeds):
    """Three generators x many sizes: tortuosity predicts A*'s gain."""
    searches = {"BFS": breadth_first_search, "DFS": depth_first_search,
                "A*": a_star_search}
    rows = []
    for generator in GENERATORS:
        for size in sizes:
            for seed in seeds:
                maze = generate_maze(size, algorithm=generator,
                                     seed=SEED + seed)
                stats = maze_stats(maze)
                for name, search in searches.items():
                    started = time.perf_counter()
                    result = search(maze)
                    rows.append({
                        "generator": generator, "size": size, "seed": seed,
                        "open_cells": stats["open_cells"],
                        "dead_ends": stats["dead_ends"],
                        "shortest": stats["shortest_path"],
                        "manhattan": stats["manhattan"],
                        "tortuosity": stats["tortuosity"],
                        "algorithm": name,
                        "nodes_expanded": result.nodes_expanded,
                        "path_length": result.path_length,
                        "seconds": time.perf_counter() - started})
    frame = pd.DataFrame(rows)
    save(frame, "scaling")
    fits = []
    for (generator, name), part in frame.groupby(["generator", "algorithm"]):
        slope, intercept = np.polyfit(np.log10(part.open_cells),
                                      np.log10(part.nodes_expanded), 1)
        fits.append({"generator": generator, "algorithm": name,
                     "slope": slope, "intercept": intercept})
    save(pd.DataFrame(fits), "scaling_fits")


@timed("E8 landmarks")
def landmarks(maze, alt16, queries):
    """Many random queries: when does landmark preprocessing pay off?"""
    rng = random.Random(SEED + 8)
    cells = [(r, c) for r, row in enumerate(maze.rows)
             for c, ch in enumerate(row) if ch != "X"]
    rows = []
    for query in range(queries):
        start, goal = rng.sample(cells, 2)
        variant = with_endpoints(maze, start=start, goal=goal)
        entry = {"query": query, "start_r": start[0], "start_c": start[1],
                 "goal_r": goal[0], "goal_c": goal[1]}
        for label, heuristic in (("manhattan", manhattan_distance),
                                 ("alt4", alt16.subset(4)),
                                 ("alt8", alt16.subset(8)),
                                 ("alt16", alt16)):
            started = time.perf_counter()
            result = a_star_search(variant, heuristic)
            entry[f"{label}_s"] = time.perf_counter() - started
            entry[f"{label}_expanded"] = result.nodes_expanded
            entry[f"{label}_path"] = result.path_length
        rows.append(entry)
    frame = pd.DataFrame(rows)
    save(frame, "landmarks")
    cumulative = np.cumsum(alt16.seconds_per_landmark)
    save(pd.DataFrame({"k": np.arange(1, len(cumulative) + 1),
                       "build_s": cumulative,
                       "row": [cell[0] for cell in alt16.landmarks],
                       "col": [cell[1] for cell in alt16.landmarks]}),
         "landmarks_build")
    update_json("summary.json",
                landmarks=[list(cell) for cell in alt16.landmarks],
                landmark_build_s=alt16.build_seconds,
                alt8_query_speedup_median=float(np.median(
                    frame.manhattan_expanded / frame.alt8_expanded)))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

STAGES = ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8")


def main(argv=None) -> int:
    """Run the selected experiments and write their output files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="small sizes for a fast smoke test")
    parser.add_argument("--maze", help="study this maze file instead of "
                        "the generated study maze")
    parser.add_argument("--size", type=int, default=1001,
                        help="size of the generated study maze")
    parser.add_argument("--only", nargs="+", choices=STAGES,
                        default=list(STAGES),
                        help="run only these experiments (results of "
                             "other stages are kept)")
    args = parser.parse_args(argv)
    quick, stages = args.quick, set(args.only)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    if args.maze:
        maze, source = load_maze(args.maze), str(args.maze)
    else:
        size = 201 if quick else args.size
        maze = generate_maze(size, algorithm="backtracker", seed=SEED)
        target = ROOT / "data" / "mazes" / (
            f"study_{size}x{size}_backtracker_seed{SEED}.txt.gz")
        target.parent.mkdir(parents=True, exist_ok=True)
        save_maze(maze, str(target))
        source = str(target.relative_to(ROOT))
    stats = maze_stats(maze)
    log(f"study maze {source}: {stats['open_cells']:,} open cells, "
        f"route {stats['shortest_path']:,}, "
        f"tortuosity {stats['tortuosity']:.1f}")
    factor = max(1, maze.width // 334)
    update_json("summary.json", study_maze=source, stats=stats,
                map_factor=factor, luck_scale=16)

    alt16 = None
    if stages & {"E1", "E2", "E8"}:
        prepared = time.perf_counter()
        alt16 = LandmarkHeuristic.farthest(maze, 16)
        log(f"16 landmarks prepared in "
            f"{time.perf_counter() - prepared:.1f} s")
    if "E1" in stages:
        benchmark(maze, alt16.subset(8), 2 if quick else 7)
    if "E2" in stages:
        heuristic_analysis(maze, alt16, factor)
    if "E3" in stages:
        census(maze, factor, 6 if quick else 30)
    if "E4" in stages:
        loops(maze, [0, 0.01, 0.1] if quick else
              [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20],
              [0] if quick else [0, 1, 2])
    if "E5" in stages:
        move_orders(maze)
    if "E6" in stages:
        open_room(maze)
    if "E7" in stages:
        texture_and_scaling([31, 61, 101] if quick else
                            [51, 101, 201, 401, 601, 801, 1001],
                            [0] if quick else [0, 1])
    if "E8" in stages:
        landmarks(maze, alt16, 5 if quick else 30)

    path = OUT / "manifest.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    seconds = {**previous.get("experiment_seconds", {}), **TIMINGS}
    update_json(
        "manifest.json",
        generated_utc=datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        mazesearch=__version__, python=platform.python_version(),
        platform=f"{platform.system()} {platform.machine()}",
        cpu=cpu_name(), numpy=np.__version__, pandas=pd.__version__,
        seed=SEED, quick=quick, experiment_seconds=seconds,
        total_seconds=round(sum(seconds.values()), 1),
        algorithms={k: v.label for k, v in ALGORITHMS.items()})
    log(f"stages {sorted(stages)} finished in "
        f"{time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
