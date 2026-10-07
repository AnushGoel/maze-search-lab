"""What-if lab: change the maze and see how each algorithm responds."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import common  # noqa: E402
from mazesearch import generate_maze, maze_to_text  # noqa: E402
from mazesearch.generate import without_interior_walls  # noqa: E402

common.page_setup("What-If Lab", "🔬",
                  "Five controlled changes to the maze, one variable at a "
                  "time.")
common.require_results()
s = common.summary()

tabs = st.tabs(["🔁 Add loops", "🎯 Move the exit", "🧭 DFS move order",
                "🏟️ Remove every wall", "🧬 Maze texture and size"])

with tabs[0]:
    loops = common.table("loops")
    measures = {"Path length / shortest": ("path_ratio", True),
                "Cells expanded (% of open cells)": ("pct_expanded", False),
                "Search time (s)": ("seconds", False)}
    choice = st.radio("Measure", list(measures), horizontal=True)
    column, log_scale = measures[choice]
    fig = go.Figure()
    for name in ["BFS", "DFS", "A*", "Bi-BFS"]:
        part = loops[loops.algorithm == name].groupby("fraction")[column]
        stats = part.agg(["mean", "min", "max"]).reset_index()
        fig.add_trace(go.Scatter(
            x=[f"{100 * f:g}%" for f in stats.fraction], y=stats["mean"],
            mode="lines+markers", name=name,
            line={"color": common.COLORS[name], "width": 2.5},
            error_y={"type": "data", "array": stats["max"] - stats["mean"],
                     "arrayminus": stats["mean"] - stats["min"]}))
    fig.update_xaxes(title_text="share of removable walls opened")
    if log_scale:
        fig.update_yaxes(type="log")
    common.chart(common.style(fig, 420, f"{choice} as loops are added "
                              "(mean of 3 seeds, bars: min-max)"))
    top = loops[loops.fraction == loops.fraction.max()]
    top = top.groupby("algorithm").mean(numeric_only=True)
    common.insight(
        f"Loops create shortcuts: with 20% of the walls opened the shortest "
        f"route falls from {common.n(s['c_star'])} to "
        f"<b>{common.n(top.loc['BFS', 'path_length'])}</b> moves. DFS stays "
        f"fastest but its route becomes <b>{top.loc['DFS', 'path_ratio']:.1f}"
        "x</b> too long, BFS still expands almost everything, and A* "
        f"drops to <b>{top.loc['A*', 'pct_expanded']:.0f}%</b> of the cells "
        "because the Manhattan distance is finally a good estimate.")

with tabs[1]:
    census = s["census"]
    cols = st.columns(4)
    cols[0].metric("Exits analysed", common.n(census["exits"]))
    cols[1].metric("Exits where DFS beats BFS",
                   f"{census['dfs_cheaper_pct']:.0f}%")
    cols[2].metric("Rank correlation with distance",
                   f"BFS {census['spearman_bfs']:.2f}",
                   f"DFS {census['spearman_dfs']:.2f}", delta_color="off")
    cols[3].metric("Census checked against real runs",
                   f"{census['census_matches']}/{census['census_checks']}",
                   "exact matches", delta_color="off")
    st.markdown(
        "Blind searches visit cells in an order that does not depend on the "
        "goal, so two complete traversals give the cost of **every** "
        "possible exit at once. The map shows, for each region, whether "
        "DFS (blue) or BFS (red) would reach an exit placed there sooner.")
    left, right = st.columns([1.15, 1])
    with left:
        luck = common.maps()["luck"].astype(float)
        luck[luck == -128] = np.nan
        luck /= s["luck_scale"]
        fig = go.Figure(go.Heatmap(
            z=np.clip(luck, -3, 3), colorscale="RdBu_r", zmid=0,
            colorbar={"title": {"text": "log2(DFS / BFS)"}},
            hovertemplate="DFS needs 2^%{z:.2f} x BFS's work"
                          "<extra></extra>"))
        fig.update_yaxes(autorange="reversed", scaleanchor="x",
                         showticklabels=False)
        fig.update_xaxes(showticklabels=False)
        common.chart(common.style(fig, 520, "DFS luck map: where would "
                                  "DFS win?", False))
    with right:
        bins = common.table("exit_bins")
        mid = np.sqrt(bins.distance_low * bins.distance_high)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mid, y=bins.dfs_q3, mode="lines",
                                 line={"width": 0}, showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=mid, y=bins.dfs_q1, mode="lines",
                                 line={"width": 0}, fill="tonexty",
                                 fillcolor="rgba(230,159,0,0.25)",
                                 name="DFS middle 50%"))
        fig.add_trace(go.Scatter(x=mid, y=bins.dfs_median, mode="lines",
                                 name="DFS median",
                                 line={"color": common.COLORS["DFS"]}))
        fig.add_trace(go.Scatter(x=mid, y=bins.bfs_median, mode="lines",
                                 name="BFS", line={"color":
                                                   common.COLORS["BFS"],
                                                   "width": 3}))
        fig.update_xaxes(type="log", title_text="true distance of the exit "
                         "from S")
        fig.update_yaxes(type="log", title_text="cells expanded")
        common.chart(common.style(fig, 520, "Cost of reaching an exit"))
    common.insight(
        "BFS's cost is a fixed function of distance (rank correlation "
        f"{census['spearman_bfs']:.2f}): near exits are cheap, far exits "
        "expensive. DFS is a lottery whose result depends on which branch "
        f"the exit sits in (correlation {census['spearman_dfs']:.2f}).")
    with st.expander("Census check against real searches"):
        sample = common.table("exits_sample")
        sample["census exact"] = ((sample.bfs == sample.census_bfs)
                                  & (sample.dfs == sample.census_dfs))
        st.dataframe(sample, hide_index=True)

with tabs[2]:
    orders = common.table("move_orders")
    bfs_work = common.table("benchmark").set_index(
        "algorithm").loc["BFS", "nodes_expanded"]
    fig = go.Figure(go.Bar(
        x=orders.order, y=orders.nodes_expanded,
        marker_color=["#E69F00" if south else "#F3D9A4"
                      for south in orders.south_before_north],
        marker_line_color=["#222" if flag else "rgba(0,0,0,0)"
                           for flag in orders.submitted],
        marker_line_width=2))
    fig.add_hline(y=bfs_work, line_dash="dash",
                  line_color=common.COLORS["BFS"], annotation_text="BFS")
    fig.update_xaxes(title_text="order in which DFS tries its moves")
    fig.update_yaxes(title_text="cells expanded")
    common.chart(common.style(fig, 420, "DFS effort for all 24 move orders "
                              "(dark = south before north; outlined = "
                              "default)", False))
    south = orders[orders.south_before_north].nodes_expanded
    north = orders[~orders.south_before_north].nodes_expanded
    common.insight(
        f"Same maze, same algorithm: the cheapest order expands "
        f"<b>{common.n(orders.nodes_expanded.min())}</b> cells and the "
        f"costliest <b>{common.n(orders.nodes_expanded.max())}</b>. Orders "
        f"that try south before north expand a median "
        f"{common.n(south.median())} cells against {common.n(north.median())}"
        " for the rest, because the exit lies to the south. That is a "
        "heuristic in disguise.")

with tabs[3]:
    room = common.table("open_room")
    st.dataframe(room.round(3), hide_index=True)

    @st.cache_data
    def room_text(size: int) -> str:
        """A small open room with the generator's start and exit."""
        return maze_to_text(without_interior_walls(
            generate_maze(size, seed=1)))

    left, right = st.columns([1, 1])
    with left:
        fig = go.Figure(go.Bar(
            x=room.algorithm, y=room.pct_expanded,
            marker_color=[common.COLORS.get(name.split(" ")[0],
                                            common.ACCENT)
                          for name in room.algorithm],
            text=[f"{v:.1f}%" for v in room.pct_expanded],
            textposition="outside"))
        fig.update_yaxes(title_text="% of open cells expanded")
        common.chart(common.style(fig, 380, "An open room is A*'s best case",
                                  False))
    with right:
        st.image(common.race_gif(room_text(31), ("bfs", "dfs", "astar")),
                 caption="The same idea in a 31 x 31 room: BFS floods, A* "
                         "walks straight, DFS depends on its move order.")
    common.insight(
        "With no walls the Manhattan distance is exact, so A* expands little "
        "more than the route itself, while BFS floods the whole room. DFS "
        "can be perfect or terrible depending only on its move order.")

with tabs[4]:
    texture = common.table("scaling")
    fits = common.table("scaling_fits")
    wide = texture.pivot_table(index=["generator", "size", "seed",
                                      "tortuosity", "open_cells"],
                               columns="algorithm",
                               values="nodes_expanded").reset_index()
    wide["A* / BFS"] = wide["A*"] / wide["BFS"]
    symbols = {"backtracker": "circle", "prim": "square",
               "binary_tree": "diamond"}
    palette = {"backtracker": "#7B2FBE", "prim": "#009E73",
               "binary_tree": "#E69F00"}
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for generator, part in wide.groupby("generator"):
            fig.add_trace(go.Scatter(
                x=part.tortuosity, y=part["A* / BFS"], mode="markers",
                name=generator, marker={"size": 11,
                                        "symbol": symbols[generator],
                                        "color": palette[generator]},
                text=[f"{size} x {size}" for size in part["size"]],
                hovertemplate="%{text}<br>tortuosity %{x:.1f}<br>A*/BFS "
                              "%{y:.2f}<extra></extra>"))
        fig.update_xaxes(type="log", title_text="tortuosity (route / "
                         "straight line)")
        fig.update_yaxes(title_text="A* work / BFS work")
        common.chart(common.style(fig, 430, "The winding-ness of a maze "
                                  "predicts A*'s gain"))
    with right:
        fig = go.Figure()
        for (generator, name), part in texture.groupby(["generator",
                                                        "algorithm"]):
            fig.add_trace(go.Scatter(
                x=part.open_cells, y=part.nodes_expanded,
                mode="markers", name=f"{name} · {generator}",
                marker={"color": common.COLORS[name],
                        "symbol": symbols[generator], "size": 8}))
        fig.update_xaxes(type="log", title_text="open cells")
        fig.update_yaxes(type="log", title_text="cells expanded")
        common.chart(common.style(fig, 430, "Work grows linearly with size"))
    st.dataframe(fits.round(3), hide_index=True)
    common.insight(
        "Every algorithm's work grows roughly linearly with the number of "
        "cells (log-log slopes near 1), but the constant depends on the "
        "maze's texture: on near-straight Prim and binary-tree mazes A* "
        "needs a small fraction of BFS's work; on winding backtracker mazes "
        "it needs nearly all of it.")
