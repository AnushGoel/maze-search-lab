"""Maze Lab: generate a maze, choose algorithms and watch them search."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import common  # noqa: E402
from mazesearch import ALGORITHMS, load_maze, render_solution  # noqa: E402
from mazesearch.render import render_maze  # noqa: E402

common.page_setup("Maze Lab", "🧪",
                  "Build a maze, pick the algorithms, and watch them "
                  "search it. Everything here runs live.")

GENERATORS = {
    "backtracker": "Recursive backtracker: long, winding corridors",
    "prim": "Prim's algorithm: short corridors, many dead ends",
    "binary_tree": "Binary tree: strong diagonal bias, near-straight routes",
}

with st.sidebar:
    st.header("1 · Maze")
    source = st.radio("Source", ["Generate", "Upload a file"],
                      horizontal=True)
    if source == "Generate":
        generator = st.selectbox("Generator", list(GENERATORS),
                                 format_func=GENERATORS.get)
        size = st.slider("Size (characters per side)", 11, 301, 61, step=2)
        loops = st.slider("Loops: removable walls opened (%)", 0, 50, 0)
        exit_mode = st.radio("Exit position", ["corner", "center", "random"],
                             horizontal=True)
        seed = int(st.number_input("Seed", min_value=0, max_value=999_999,
                                   value=7, step=1))
        text = common.make_maze_text(generator, size, seed, loops / 100,
                                     exit_mode)
    else:
        upload = st.file_uploader("Maze file: S start, E exit, X walls",
                                  type=["txt"])
        if upload is None:
            st.info("Upload a maze to continue.")
            st.stop()
        text = upload.getvalue().decode("utf-8", "replace")
    st.header("2 · Algorithms")
    keys = st.multiselect("Run", list(ALGORITHMS),
                          default=["bfs", "dfs", "astar"],
                          format_func=lambda key: ALGORITHMS[key].label)
    st.download_button("Download this maze", text, file_name="maze.txt")

if not keys:
    st.warning("Pick at least one algorithm in the sidebar.")
    st.stop()
try:
    maze = load_maze(text)
except ValueError as error:
    st.error(f"That maze is not valid: {error}")
    st.stop()
if maze.width * maze.height > 400_000:
    st.error("The live lab is limited to about 600 x 600 cells; use the "
             "command line for larger mazes.")
    st.stop()

stats, runs = common.run_searches(text, tuple(keys))
route = stats["shortest_path"]
cols = st.columns(5)
cols[0].metric("Open cells", common.n(stats["open_cells"]))
cols[1].metric("Dead ends", common.n(stats["dead_ends"]))
cols[2].metric("Loops", common.n(stats["loops"]))
cols[3].metric("Shortest route", "none" if route is None else common.n(route))
cols[4].metric("Tortuosity", "-" if route is None
               else f"{stats['tortuosity']:.1f}x",
               help="Shortest route divided by the straight-line "
                    "(Manhattan) distance from S to E.")
if route is None:
    st.warning("E cannot be reached from S: every algorithm must explore "
               "the whole reachable region before reporting failure.")

rows = []
for run in runs:
    result = run["result"]
    rows.append({
        "Algorithm": run["label"],
        "Family": ALGORITHMS[run["key"]].family,
        "Path length": result.path_length,
        "x shortest": (None if result.path is None or not route
                       else round(result.path_length / route, 2)),
        "Cells expanded": result.nodes_expanded,
        "% of open cells": round(100 * result.nodes_expanded
                                 / stats["open_cells"], 1),
        "Peak frontier": result.max_frontier,
        "Time (ms, traced)": round(1000 * run["seconds"], 1),
    })
frame = pd.DataFrame(rows)
st.dataframe(frame, hide_index=True)

left, right = st.columns(2)
colors = [common.COLORS.get(label, common.ACCENT) for label in frame.Algorithm]
with left:
    fig = go.Figure(go.Bar(x=frame.Algorithm, y=frame["Cells expanded"],
                           marker_color=colors,
                           text=[common.n(v) for v in frame["Cells expanded"]],
                           textposition="outside"))
    common.chart(common.style(fig, 340, "Cells expanded (work)", False))
with right:
    fig = go.Figure(go.Bar(x=frame.Algorithm, y=frame["x shortest"],
                           marker_color=colors,
                           text=frame["x shortest"], textposition="outside"))
    fig.add_hline(y=1, line_dash="dot", line_color="gray")
    common.chart(common.style(fig, 340, "Path length / shortest (quality)",
                              False))

labels = list(frame.Algorithm)
if "BFS" in labels and "A*" in labels and route:
    bfs = frame.set_index("Algorithm").loc["BFS", "Cells expanded"]
    astar = frame.set_index("Algorithm").loc["A*", "Cells expanded"]
    common.insight(
        f"On this maze A* expanded <b>{100 * (1 - astar / bfs):.0f}% fewer"
        f"</b> cells than BFS. The route is <b>{stats['tortuosity']:.1f}x"
        "</b> the straight-line distance: the higher that number, the less "
        "the Manhattan heuristic can help. Try the Prim generator or add "
        "loops and watch the gap grow.")

st.subheader("Where each algorithm searched")
st.caption("Colour = order of expansion (dark early, yellow late); white = "
           "never expanded; red = route; green S, gold E.")
scale = max(1, 420 // maze.width)
for start in range(0, len(runs), 4):
    columns = st.columns(4)
    for column, run in zip(columns, runs[start:start + 4]):
        image = render_maze(maze, run["order"], run["result"].path,
                            scale=scale)
        column.image(image, caption=run["label"])

st.subheader("Replay")
longest = max(run["result"].nodes_expanded for run in runs)
if maze.width <= 151 and st.toggle("Animated race (GIF)", value=True):
    st.image(common.race_gif(text, tuple(keys)),
             caption="All panels share one clock in expanded cells; a "
                     "panel stops when its search finishes.")
step = st.slider("Step through the search (cells expanded so far)", 0,
                 max(longest, 1), max(longest // 3, 1))
columns = st.columns(min(len(runs), 4))
for column, run in zip(columns, runs[:4]):
    done = step >= run["result"].nodes_expanded
    image = render_maze(maze, run["order"],
                        run["result"].path if done else None,
                        upto=step, scale=scale)
    column.image(image, caption=f"{run['label']}: "
                 f"{min(step, run['result'].nodes_expanded):,} expanded"
                 + (" (done)" if done else ""))

with st.expander("Download the solved mazes"):
    for run in runs:
        if run["result"].path is not None:
            st.download_button(f"{run['label']} solution",
                               render_solution(maze, run["result"].path),
                               file_name=f"solution_{run['key']}.txt",
                               key=f"download_{run['key']}")
