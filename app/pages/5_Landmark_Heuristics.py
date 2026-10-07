"""Landmark heuristics (ALT): fixing A* with better information."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import common  # noqa: E402

common.page_setup("Landmark Heuristics", "📍",
                  "A better heuristic, not a better algorithm: exact "
                  "distances to a few landmarks make A* far more "
                  "selective.")
common.require_results()

s = common.summary()
marks = common.table("landmarks")
build = common.table("landmarks_build").set_index("k")

st.markdown(
    "The ALT method (A*, Landmarks and the Triangle inequality) of "
    "Goldberg and Harrelson (2005) precomputes exact distances from a few "
    "landmark cells to every cell. For any landmark L, the triangle "
    "inequality bounds the distance from n to the target t, and the best "
    "bound over all landmarks is still admissible and consistent, so A* "
    "stays optimal. Landmarks are chosen by farthest-point selection.")
st.latex(r"h_{\mathrm{ALT}}(n) = \max_{L}\,\bigl|\,d(L, n) - d(L, t)\,"
         r"\bigr| \;\le\; d(n, t)")

slow = marks.manhattan_s.mean()
speed = {k: (marks.manhattan_expanded / marks[f"alt{k}_expanded"]).median()
         for k in (4, 8, 16)}
cols = st.columns(4)
cols[0].metric("Fewer cells expanded (k = 8)", f"{speed[8]:.1f}x",
               "median over random queries", delta_color="off")
cols[1].metric("Faster queries (k = 8)",
               f"{(marks.manhattan_s / marks.alt8_s).median():.1f}x",
               "median wall-clock speed-up", delta_color="off")
cols[2].metric("Preparation for 8 landmarks",
               f"{build.loc[8, 'build_s']:.1f} s", "one BFS per landmark",
               delta_color="off")
queries = common.break_even(build.loc[8, "build_s"], slow,
                            marks.alt8_s.mean())
cols[3].metric("Pays for itself after", f"{queries:.0f} queries")

same = bool((marks.manhattan_path == marks.alt8_path).all()
            and (marks.manhattan_path == marks.alt16_path).all())
st.success(f"Optimality check: Manhattan and landmark A* returned routes of "
           f"identical length on all {len(marks)} random queries."
           if same else "Warning: some route lengths differ.")

left, right = st.columns([1, 1.2])
with left:
    k = st.slider("Landmarks to show", 1, len(build), 8)
    maps = common.maps()
    distance = common.decode(maps["distance"])
    factor = s["map_factor"]
    fig = go.Figure(go.Heatmap(z=distance, colorscale="Viridis",
                               showscale=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=build.col.iloc[:k] / factor, y=build.row.iloc[:k] / factor,
        mode="markers+text", text=[str(i) for i in build.index[:k]],
        textposition="top center", textfont={"color": "white"},
        marker={"size": 14, "color": "#FF3860", "symbol": "star",
                "line": {"color": "white", "width": 1}},
        hovertemplate="landmark %{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", scaleanchor="x",
                     showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    common.chart(common.style(fig, 470, "Landmarks on the distance field",
                              False))
with right:
    fig = go.Figure()
    for count, color in ((4, "#B39DDB"), (8, common.ACCENT),
                         (16, "#3D1A78")):
        ratio = marks.manhattan_expanded / marks[f"alt{count}_expanded"]
        fig.add_trace(go.Box(y=ratio, name=f"k = {count}",
                             marker_color=color, boxpoints="all",
                             jitter=0.4, pointpos=0))
    fig.add_hline(y=1, line_dash="dot", line_color="gray")
    fig.update_yaxes(type="log", title_text="Manhattan work / landmark work")
    common.chart(common.style(fig, 470, f"Work saved on {len(marks)} "
                              "random queries", False))

st.subheader("When does the preparation pay off?")
horizon = np.arange(0, max(60, int(3 * min(queries, 200))) + 1)
fig = go.Figure(go.Scatter(x=horizon, y=horizon * slow, mode="lines",
                           name="A* + Manhattan",
                           line={"color": common.COLORS["A*"], "width": 3}))
for count, color in ((4, "#B39DDB"), (8, common.ACCENT), (16, "#3D1A78")):
    fast = marks[f"alt{count}_s"].mean()
    fig.add_trace(go.Scatter(
        x=horizon, y=build.loc[count, "build_s"] + horizon * fast,
        mode="lines", name=f"A* + ALT, k = {count} (incl. preparation)",
        line={"color": color, "width": 2.5}))
fig.update_xaxes(title_text="number of queries on the same maze")
fig.update_yaxes(title_text="total seconds")
common.chart(common.style(fig, 420, "Cumulative time"))
common.insight(
    f"With 8 landmarks A* needed a median <b>{speed[8]:.1f}x</b> fewer "
    f"expansions per query and every route stayed optimal. Preparation "
    f"costs <b>{build.loc[8, 'build_s']:.1f} s</b>, so it pays off after "
    f"about <b>{queries:.0f}</b> queries; for a single query, plain A* is "
    "still the cheaper choice. The lesson matches road-network routing, "
    "where geometric bounds are weak and landmark bounds are the standard "
    "fix (Bast et al., 2016).")
with st.expander("Per-query results"):
    st.dataframe(marks, hide_index=True)
