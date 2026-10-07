"""Benchmark: eight algorithms on the 1,001 x 1,001 study maze."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import common  # noqa: E402

common.page_setup("Benchmark", "⏱️",
                  "One large perfect maze, eight algorithms: how long they "
                  "take, how much they explore and how much memory they "
                  "need.")
common.require_results()

s = common.summary()
bench = common.table("benchmark")
runs = common.table("benchmark_runs")
by_name = bench.set_index("algorithm")
stats = s["stats"]

cols = st.columns(4)
cols[0].metric("Open cells", common.n(stats["open_cells"]))
cols[1].metric("Shortest route", common.n(s["c_star"]),
               f"{stats['tortuosity']:.1f}x the straight line",
               delta_color="off")
fastest = by_name.median_s.idxmin()
cols[2].metric("Fastest search", fastest,
               f"{by_name.loc[fastest, 'median_s']:.2f} s median",
               delta_color="off")
least = by_name.nodes_expanded.idxmin()
cols[3].metric("Least work", least,
               f"{common.n(by_name.loc[least, 'nodes_expanded'])} cells",
               delta_color="off")

core = ["BFS", "DFS", "A*"]
show_all = st.toggle("Include the five extra algorithms", value=True)
view = bench if show_all else bench[bench.algorithm.isin(core)]
colors = [common.COLORS.get(name, common.ACCENT) for name in view.algorithm]

tab_time, tab_work, tab_memory, tab_table = st.tabs(
    ["⏱️ Time", "🔍 Work", "💾 Memory", "📋 Full table"])

with tab_time:
    left, right = st.columns([1.2, 1])
    with left:
        fig = go.Figure(go.Bar(
            x=view.algorithm, y=view.median_s, marker_color=colors,
            error_y={"type": "data", "array": view.q3_s - view.median_s,
                     "arrayminus": view.median_s - view.q1_s},
            text=[f"{v:.2f} s" for v in view.median_s],
            textposition="outside"))
        fig.update_yaxes(title_text="seconds")
        common.chart(common.style(fig, 390, "Median search time (whiskers: "
                                  "interquartile range)", False))
    with right:
        fig = go.Figure()
        for name in view.algorithm:
            part = runs[runs.algorithm == name]
            fig.add_trace(go.Box(y=part.seconds, name=name,
                                 marker_color=common.COLORS.get(
                                     name, common.ACCENT),
                                 boxpoints="all", jitter=0.5, pointpos=0))
        fig.update_yaxes(title_text="seconds")
        common.chart(common.style(fig, 390, "Every timed run", False))
    fig = go.Figure(go.Bar(x=view.algorithm, y=view.us_per_expansion,
                           marker_color=colors,
                           text=[f"{v:.2f}" for v in view.us_per_expansion],
                           textposition="outside"))
    fig.update_yaxes(title_text="microseconds per expanded cell")
    common.chart(common.style(fig, 320, "Cost of one expansion", False))
    bfs, astar = by_name.loc["BFS"], by_name.loc["A*"]
    common.insight(
        f"Time = work x cost per expansion. A* did "
        f"<b>{100 * astar.nodes_expanded / bfs.nodes_expanded:.1f}%</b> of "
        f"BFS's work, but each expansion cost "
        f"<b>{astar.us_per_expansion / bfs.us_per_expansion:.1f}x</b> as "
        "much (a heap push and a heuristic call for every new cell), so it "
        f"finished <b>{astar.median_s / bfs.median_s:.1f}x slower</b>.")

with tab_work:
    left, right = st.columns([1.2, 1])
    with left:
        fig = go.Figure(go.Bar(
            x=view.algorithm, y=view.nodes_expanded, marker_color=colors,
            text=[f"{v:.0f}%" for v in view.pct_open_expanded],
            textposition="outside"))
        fig.add_hline(y=stats["open_cells"], line_dash="dot",
                      line_color="gray",
                      annotation_text="all open cells")
        fig.update_yaxes(title_text="cells expanded")
        common.chart(common.style(fig, 400, "Cells expanded (labels: % of "
                                  "the maze)", False))
    with right:
        profile = s["profiles"]["Manhattan"]
        stages = ["Open cells", "Closer to S than E is",
                  "Expanded by BFS", "Expanded by A*",
                  "Expanded by A* + landmarks", "On the route"]
        values = [stats["open_cells"], profile["closer_than_exit"],
                  by_name.loc["BFS", "nodes_expanded"],
                  by_name.loc["A*", "nodes_expanded"],
                  by_name.loc["A* + ALT (k=8)", "nodes_expanded"],
                  s["c_star"] + 1]
        fig = go.Figure(go.Funnel(
            y=stages, x=values, textinfo="value+percent initial",
            marker={"color": ["#9AA0A6", "#B39DDB", common.COLORS["BFS"],
                              common.COLORS["A*"], common.ACCENT,
                              "#FF3860"]}))
        common.chart(common.style(fig, 400, "Where the work goes", False))
    common.insight(
        "BFS must expand essentially every cell that is closer to S than "
        "E is, and on this maze that is almost all of them. Everything "
        "between that bar and the route is work a better heuristic could "
        "avoid; the landmark heuristic recovers a large part of it.")

with tab_memory:
    left, right = st.columns(2)
    with left:
        fig = go.Figure(go.Bar(x=view.algorithm, y=view.peak_memory_mib,
                               marker_color=colors,
                               text=[f"{v:.0f}" for v in
                                     view.peak_memory_mib],
                               textposition="outside"))
        fig.update_yaxes(title_text="MiB (tracemalloc peak)")
        common.chart(common.style(fig, 380, "Peak memory", False))
    with right:
        fig = go.Figure(go.Bar(x=view.algorithm, y=view.peak_frontier,
                               marker_color=colors,
                               text=[common.n(v) for v in view.peak_frontier],
                               textposition="outside"))
        fig.update_yaxes(type="log", title_text="entries (log scale)")
        common.chart(common.style(fig, 380, "Largest frontier", False))
    common.insight(
        f"The textbook says BFS needs a huge frontier. Here its queue never "
        f"held more than <b>{common.n(by_name.loc['BFS', 'peak_frontier'])}"
        f"</b> cells, against <b>"
        f"{common.n(by_name.loc['DFS', 'peak_frontier'])}</b> for DFS's "
        "stack: a perfect maze is deep and narrow, so each distance ring is "
        "tiny. The table of reached cells, not the frontier, dominates "
        "memory.")

with tab_table:
    st.dataframe(view.round(4), hide_index=True)
    st.caption("Times are medians of interleaved runs after a warm-up, "
               "measured around the search call only. Memory comes from "
               "separate tracemalloc runs.")

st.subheader("Where each algorithm searched")
st.caption("Downsampled maps of the study maze. Colour = expansion order "
           "(dark early, yellow late); white = never expanded.")
maps = common.maps()
columns = st.columns(4)
for column, (key, label) in zip(columns, [
        ("order_BFS", "BFS"), ("order_DFS", "DFS"), ("order_A*", "A*"),
        ("order_ALT8", "A* + landmarks (k=8)")]):
    column.image(common.colorize(common.decode(maps[key]), zoom=1),
                 caption=label)
