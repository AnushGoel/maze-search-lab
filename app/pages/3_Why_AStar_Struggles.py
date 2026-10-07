"""Why A* struggles: the pruning band and the quality of the heuristic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import common  # noqa: E402

common.page_setup("Why A* Struggles", "🧠",
                  "A heuristic can only skip cells it can prove are off "
                  "the shortest route. On a winding maze there are almost "
                  "none.")
common.require_results()

s = common.summary()
profile = s["profiles"]["Manhattan"]
c_star, h_max = s["c_star"], profile["h_max"]
hist = common.table("depth_hist")
route = common.table("route_profile")
curve = common.table("quality_curve")
heur = common.table("heuristics")

st.markdown(
    "A* with a consistent heuristic expands every cell whose estimated "
    "total cost is below the cost of the optimal route C* (Hart et al., "
    "1968; Dechter & Pearl, 1985). The Manhattan distance can never exceed "
    "the width plus the height of the grid, so every cell whose true "
    "distance from S is below C* minus that ceiling must be expanded, "
    "whatever the heuristic says.")
st.latex(r"g^*(n) + h(n) < C^* \;\Longrightarrow\; n \text{ is expanded}"
         r"\qquad\text{and}\qquad h(n) \le h_{\max}")

cols = st.columns(4)
cols[0].metric("Optimal cost C*", common.n(c_star))
cols[1].metric("Largest Manhattan value", common.n(h_max))
cols[2].metric("Cells closer to S than E", common.n(
    profile["closer_than_exit"]), "what BFS must expand", delta_color="off")
cols[3].metric("Cells A* can skip", common.n(profile["prunable"]),
               f"{100 * profile['prunable'] / profile['reachable']:.2f}% "
               "of the maze", delta_color="off")

left, right = st.columns(2)
with left:
    centers = (hist.bin_start + hist.bin_end) / 2
    fig = go.Figure(go.Bar(x=centers, y=hist.cells, marker_color="#9AA0A6",
                           name="cells"))
    fig.add_vrect(x0=max(c_star - h_max, 0), x1=c_star,
                  fillcolor=common.COLORS["A*"], opacity=0.35, line_width=0)
    fig.add_vline(x=c_star, line_color=common.COLORS["A*"],
                  annotation_text=f"C* = {c_star:,}",
                  annotation_position="top left")
    fig.update_xaxes(title_text="true distance from S, g*(n)")
    fig.update_yaxes(title_text="open cells")
    common.chart(common.style(fig, 400, "Only the green band is prunable",
                              False))
with right:
    fig = go.Figure()
    for column, label, color in (
            ("h_true", "true remaining cost h*(n)", "#3C4043"),
            ("alt8", "landmark bound (k = 8)", common.ACCENT),
            ("manhattan", "Manhattan distance", common.COLORS["A*"])):
        fig.add_trace(go.Scatter(x=route.position, y=route[column],
                                 mode="lines", name=label,
                                 line={"color": color, "width": 2.5}))
    fig.update_xaxes(title_text="position on the route (moves from S)")
    fig.update_yaxes(title_text="moves remaining to E")
    common.chart(common.style(fig, 400, "Estimate versus truth along the "
                              "route"))

along = (route.manhattan / route.h_true.where(route.h_true > 0)).mean()
common.insight(
    f"Along the route the Manhattan distance captures on average only "
    f"<b>{100 * along:.1f}%</b> of the true remaining cost. The estimate is "
    "admissible and consistent, which guarantees an optimal route, but it "
    "carries almost no information about which branch leads to E.")

st.subheader("How good must a heuristic be?")
left, right = st.columns([1, 1.3])
with left:
    quality = st.slider("Heuristic quality q: h = (1 - q) x Manhattan + "
                        "q x true distance", 0.0, 1.0, 0.5, 0.05)
    row = curve.iloc[(curve.quality - quality).abs().argmin()]
    st.metric("Cells A* must expand at this quality",
              common.n(row.must_expand),
              f"{100 * row.must_expand / profile['must_expand']:.0f}% of "
              "the Manhattan figure", delta_color="off")
    labels = ["Manhattan", "ALT k=4", "ALT k=8", "ALT k=16", "Exact h*"]
    must = [s["profiles"][label]["must_expand"] for label in labels]
    fig = go.Figure(go.Bar(x=labels, y=must,
                           marker_color=[common.COLORS["A*"]]
                           + [common.ACCENT] * 3 + ["#3C4043"],
                           text=[common.n(v) for v in must],
                           textposition="outside"))
    common.chart(common.style(fig, 300, "Cells A* is forced to expand",
                              False))
with right:
    fig = go.Figure(go.Scatter(x=curve.quality, y=curve.must_expand,
                               mode="lines+markers", line={"width": 3,
                                                           "color":
                                                           common.ACCENT}))
    fig.add_trace(go.Scatter(x=[row.quality], y=[row.must_expand],
                             mode="markers", marker={"size": 16,
                                                     "color": "#FF3860"}))
    fig.update_xaxes(title_text="quality q (0 = Manhattan, 1 = exact)")
    fig.update_yaxes(title_text="cells that must be expanded")
    common.chart(common.style(fig, 430, "Blending the heuristic toward "
                              "the truth", False))
common.insight(
    "Admissible blends of the Manhattan distance and the true distance "
    "shrink the forced work smoothly. The landmark heuristic reaches the "
    "far end of this curve without knowing the true distance in advance.")

st.subheader("The maze's distance field")
maps = common.maps()
distance = common.decode(maps["distance"]) * hist.bin_end.max()
fig = go.Figure(go.Heatmap(z=distance, colorscale="Viridis",
                           colorbar={"title": {"text": "moves from S"}},
                           hovertemplate="distance from S ~ %{z:,.0f}"
                                         "<extra></extra>"))
fig.update_yaxes(autorange="reversed", scaleanchor="x", showticklabels=False)
fig.update_xaxes(showticklabels=False)
common.chart(common.style(fig, 560, "Every cell coloured by its true "
                          "distance from S (top left)", False))

st.subheader("Same A*, different heuristics")
fig = go.Figure(go.Bar(x=heur.heuristic, y=heur.pct_of_bfs,
                       marker_color=[common.ACCENT if h.startswith("ALT")
                                     else common.COLORS["A*"]
                                     for h in heur.heuristic],
                       text=[f"{v:.1f}%" for v in heur.pct_of_bfs],
                       textposition="outside"))
fig.update_yaxes(title_text="cells expanded, % of BFS")
common.chart(common.style(fig, 380, "Work relative to BFS", False))
st.dataframe(heur.round(3), hide_index=True)
st.caption("Weighted and greedy variants are not consistent, so the "
           "forced-expansion count does not apply to them.")
