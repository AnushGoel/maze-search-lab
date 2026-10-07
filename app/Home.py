"""Maze Search Lab: an interactive look at blind and heuristic search.

Run with ``streamlit run app/Home.py`` from the project folder.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

import common  # noqa: E402

common.page_setup(
    "Maze Search Lab", "🧭",
    "How much does a heuristic really help? Breadth-first, depth-first "
    "and A* search on mazes of up to a million cells.")

left, right = st.columns([1.1, 1], gap="large")
with left:
    st.markdown(
        "#### The question\n"
        "A* is taught as the clever way to find a path: a heuristic "
        "estimate of the remaining distance steers it toward the goal, so "
        "it should explore far less than a blind search. This project "
        "tests that promise on large mazes, measures *why* it sometimes "
        "fails, and shows what fixes it.\n\n"
        "#### What is in here\n"
        "Every number in the app comes from reproducible, seeded "
        "experiments (`scripts/run_experiments.py`) run with the "
        "`mazesearch` package. The **Maze Lab** runs the algorithms live "
        "on mazes you generate or upload; the other pages walk through "
        "the study in order.")
with right:
    demo = common.make_maze_text("backtracker", 41, 7, 0.0, "corner")
    st.image(common.race_gif(demo, ("bfs", "dfs", "astar")),
             caption="One 41 x 41 maze, one shared clock (cells expanded). "
                     "Colour shows when a cell was expanded; red is the "
                     "route.")

if common.has_results():
    s = common.summary()
    bench = common.table("benchmark").set_index("algorithm")
    loops = common.table("loops")
    texture = common.table("scaling")
    marks = common.table("landmarks")
    build = common.table("landmarks_build").set_index("k")
    bfs, astar = bench.loc["BFS"], bench.loc["A*"]
    saved = 100 * (1 - astar.nodes_expanded / bfs.nodes_expanded)
    st.divider()
    cols = st.columns(4)
    cols[0].metric("Open cells in the study maze",
                   common.n(s["stats"]["open_cells"]))
    cols[1].metric("Shortest route (moves)", common.n(s["c_star"]),
                   f"{s['stats']['tortuosity']:.0f}x the straight line",
                   delta_color="off")
    cols[2].metric("Work A* saves over BFS", f"{saved:.1f}%",
                   f"{astar.median_s / bfs.median_s:.1f}x BFS's time",
                   delta_color="off")
    cols[3].metric("Landmark A* (8 landmarks)",
                   f"{s['alt8_query_speedup_median']:.1f}x fewer",
                   "cells expanded per query (median)", delta_color="off")

    st.subheader("Five findings, in the order the app tells them")
    top20 = loops[loops.fraction == loops.fraction.max()]
    top20 = top20.groupby("algorithm").mean(numeric_only=True)
    tort = texture.groupby("generator").tortuosity.median()
    ratio = (texture.pivot_table(index=["generator", "size", "seed"],
                                 columns="algorithm",
                                 values="nodes_expanded"))
    gain = (ratio["A*"] / ratio["BFS"]).groupby("generator").median()
    slow = marks.manhattan_s.mean()
    fast = marks.alt8_s.mean()
    queries = common.break_even(build.loc[8, "build_s"], slow, fast)
    findings = [
        ("Benchmark", "A* barely beats BFS on a winding maze",
         f"It expanded {common.n(astar.nodes_expanded)} cells against "
         f"{common.n(bfs.nodes_expanded)} for BFS and took "
         f"{astar.median_s / bfs.median_s:.1f} times as long.",
         "pages/2_Benchmark.py"),
        ("Why A* struggles", "The heuristic can only prune a thin band",
         f"Only {common.n(s['profiles']['Manhattan']['prunable'])} cells "
         "are ever skippable with the Manhattan distance, because the "
         f"route is {s['stats']['tortuosity']:.0f} times longer than the "
         "straight line.", "pages/3_Why_AStar_Struggles.py"),
        ("What-if: loops", "Loops break DFS and unlock A*",
         f"With 20% of walls opened, DFS's route is "
         f"{top20.loc['DFS', 'path_ratio']:.1f}x the shortest, while A* "
         f"expands {top20.loc['A*', 'pct_expanded']:.0f}% of the cells "
         f"against {top20.loc['BFS', 'pct_expanded']:.0f}% for BFS.",
         "pages/4_What_If_Lab.py"),
        ("What-if: texture", "Maze texture decides the winner",
         f"Backtracker mazes have routes {tort['backtracker']:.0f}x the "
         f"straight line and A* needs {100 * gain['backtracker']:.0f}% of "
         f"BFS's work; on Prim mazes (tortuosity {tort['prim']:.1f}) it "
         f"needs {100 * gain['prim']:.0f}%.", "pages/4_What_If_Lab.py"),
        ("Extension", "Landmarks fix the heuristic",
         f"Exact distances to 8 landmarks cut A*'s work by a median "
         f"{s['alt8_query_speedup_median']:.1f}x per query; the "
         f"{build.loc[8, 'build_s']:.1f} s of preparation pays off after "
         f"about {queries:.0f} queries.",
         "pages/5_Landmark_Heuristics.py"),
    ]
    rows = [st.columns(3), st.columns(3)]
    slots = rows[0] + rows[1][:2]
    for slot, (kicker, headline, body, page) in zip(slots, findings):
        with slot:
            with st.container(border=True):
                common.card(kicker, headline, body)
                st.page_link(page, label="Open the evidence", icon="➡️")
    with rows[1][2]:
        with st.container(border=True):
            common.card("Try it yourself", "Run the algorithms live",
                        "Generate any maze, add loops, move the exit and "
                        "watch the searches race.")
            st.page_link("pages/1_Maze_Lab.py", label="Open the Maze Lab",
                         icon="🧪")
else:
    st.info("Run `python scripts/run_experiments.py` to create the study "
            "results; the Maze Lab works without them.")

st.divider()
st.caption("Maze Search Lab · built by Anush Goel · code in the "
           "`mazesearch` package · data in `data/results` · methods and "
           "references on the last page.")
