"""Methods, environment, data dictionary and references."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import common  # noqa: E402

common.page_setup("Methods & Data", "📚",
                  "How the numbers were produced, where they live, and "
                  "how to reproduce them.")
common.require_results()

env = common.manifest()
s = common.summary()

st.subheader("Measurement protocol")
st.markdown(
    "- **Study maze.** A 1,001 x 1,001 recursive-backtracker maze "
    f"(`{s['study_maze']}`), seed {env['seed']}, S top-left, E "
    "bottom-right.\n"
    "- **Time** is measured around the search call only "
    "(`time.perf_counter`). Each algorithm gets one warm-up run, then the "
    "timed runs rotate through the algorithms so background load hits all "
    "of them equally; medians and interquartile ranges are reported.\n"
    "- **Memory** is the peak of Python allocations during one search "
    "(`tracemalloc`), measured in separate runs because tracing slows "
    "execution.\n"
    "- **Work** (cells expanded, frontier size, path length) is exact and "
    "machine-independent, so it is the primary evidence; timings are "
    "secondary.\n"
    "- **Correctness.** Every optimal algorithm is checked against BFS on "
    "random mazes with loops, and the exit census is checked against real "
    "searches.")

left, right = st.columns(2)
with left:
    st.subheader("Environment")
    st.dataframe(pd.DataFrame({"item": ["Generated (UTC)", "Python", "CPU",
                                        "Platform", "NumPy", "pandas",
                                        "mazesearch", "Quick mode"],
                               "value": [str(env["generated_utc"]),
                                         env["python"], env["cpu"],
                                         env["platform"], env["numpy"],
                                         env["pandas"], env["mazesearch"],
                                         str(env["quick"])]}),
                 hide_index=True)
with right:
    st.subheader("Experiment run times")
    st.dataframe(pd.DataFrame(sorted(env["experiment_seconds"].items()),
                              columns=["experiment", "seconds"]),
                 hide_index=True)

st.subheader("Data dictionary")
DESCRIPTIONS = {
    "benchmark.csv": "One row per algorithm: path, work, memory, time.",
    "benchmark_runs.csv": "Every individual timed run.",
    "heuristics.csv": "A* with nine heuristics: work and forced cells.",
    "depth_hist.csv": "Histogram of true distances from S.",
    "route_profile.csv": "h*, Manhattan and landmark bound along the route.",
    "quality_curve.csv": "Forced expansions as the heuristic improves.",
    "exit_bins.csv": "Exit census summarised by distance from S.",
    "exits_sample.csv": "Census values checked against real searches.",
    "loops.csv": "Every run of the loop (braiding) experiment.",
    "move_orders.csv": "DFS work for all 24 move orders.",
    "open_room.csv": "The maze with every interior wall removed.",
    "scaling.csv": "Three generators x seven sizes x two seeds.",
    "scaling_fits.csv": "Log-log slopes of work against maze size.",
    "landmarks.csv": "Random queries: Manhattan versus landmark A*.",
    "landmarks_build.csv": "Landmark positions and preparation time.",
    "maps.npz": "Downsampled maps for the app (uint8 / int8).",
    "summary.json": "Headline numbers used across the app.",
    "manifest.json": "Environment, seed and run times.",
}
rows = []
for path in sorted(common.RESULTS.iterdir()):
    entry = {"file": path.name, "kB": round(path.stat().st_size / 1024, 1),
             "contents": DESCRIPTIONS.get(path.name, "")}
    if path.suffix == ".csv":
        frame = pd.read_csv(path)
        entry["rows x columns"] = f"{len(frame)} x {frame.shape[1]}"
    rows.append(entry)
st.dataframe(pd.DataFrame(rows), hide_index=True)

st.subheader("Reproduce everything")
st.code("pip install -r requirements.txt\n"
        "python scripts/run_experiments.py      # about 8 minutes\n"
        "python -m pytest                       # tests\n"
        "streamlit run app/Home.py              # this app", language="bash")

st.subheader("References")
st.markdown(
    "Bast, H., Delling, D., Goldberg, A., Müller-Hannemann, M., Pajor, T., "
    "Sanders, P., Wagner, D., & Werneck, R. F. (2016). Route planning in "
    "transportation networks. In L. Kliemann & P. Sanders (Eds.), "
    "*Algorithm engineering: Selected results and surveys* (pp. 19–80). "
    "Springer. https://doi.org/10.1007/978-3-319-49487-6_2\n\n"
    "Buck, J. (2015). *Mazes for programmers: Code your own twisty little "
    "passages*. Pragmatic Bookshelf.\n\n"
    "Dechter, R., & Pearl, J. (1985). Generalized best-first search "
    "strategies and the optimality of A*. *Journal of the ACM, 32*(3), "
    "505–536. https://doi.org/10.1145/3828.3830\n\n"
    "Goldberg, A. V., & Harrelson, C. (2005). Computing the shortest path: "
    "A* search meets graph theory. In *Proceedings of the Sixteenth Annual "
    "ACM-SIAM Symposium on Discrete Algorithms* (pp. 156–165). Society for "
    "Industrial and Applied Mathematics.\n\n"
    "Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for "
    "the heuristic determination of minimum cost paths. *IEEE Transactions "
    "on Systems Science and Cybernetics, 4*(2), 100–107. "
    "https://doi.org/10.1109/TSSC.1968.300136\n\n"
    "Russell, S., & Norvig, P. (2021). *Artificial intelligence: A modern "
    "approach* (4th ed.). Pearson.\n\n"
    "The full reference list is in `docs/RESEARCH.md`.")
