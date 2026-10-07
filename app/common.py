"""Shared helpers for the Streamlit app: paths, data, charts and images."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
RESULTS = ROOT / "data" / "results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mazesearch import ALGORITHMS, load_maze, maze_to_text  # noqa: E402
from mazesearch.analysis import expansion_order, maze_stats  # noqa: E402
from mazesearch.generate import (braid, generate_maze,  # noqa: E402
                                 pick_cell, with_endpoints)
from mazesearch.render import (as_grid, gradient, race_frames,  # noqa: E402
                               to_gif)

# One colour per algorithm, everywhere in the app (Okabe-Ito palette).
COLORS = {
    "BFS": "#0072B2", "DFS": "#E69F00", "A*": "#009E73",
    "Bi-BFS": "#56B4E9", "UCS": "#8C8C8C", "WA* (w=2)": "#CC79A7",
    "Greedy": "#D55E00", "A* + ALT (k=8)": "#7B2FBE",
}
ACCENT = "#7B2FBE"

_CSS = """
<style>
div[data-testid="stMetricValue"] { font-size: 1.55rem; }
.insight { border-left: 4px solid #7B2FBE; padding: .75rem 1rem;
           background: rgba(123, 47, 190, .07); border-radius: .4rem;
           margin: .3rem 0 1rem 0; line-height: 1.55; }
.kicker { text-transform: uppercase; letter-spacing: .08em;
          font-size: .72rem; opacity: .7; margin-bottom: .2rem; }
.big { font-size: 1.15rem; font-weight: 650; margin-bottom: .35rem; }
</style>
"""


def page_setup(title: str, icon: str, subtitle: str) -> None:
    """Configure the page and draw a consistent header."""
    st.set_page_config(page_title=f"{title} | Maze Search Lab",
                       page_icon=icon, layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)
    st.title(f"{icon} {title}")
    st.caption(subtitle)


def insight(html: str) -> None:
    """A highlighted call-out box for the key takeaway of a section."""
    st.markdown(f'<div class="insight">💡 {html}</div>',
                unsafe_allow_html=True)


def card(kicker: str, headline: str, body: str) -> None:
    """A short finding: small label, bold headline, one sentence."""
    st.markdown(f'<div class="kicker">{kicker}</div>'
                f'<div class="big">{headline}</div>{body}',
                unsafe_allow_html=True)


def n(value: float) -> str:
    """Format a count with thousands separators."""
    return f"{value:,.0f}"


# ---------------------------------------------------------------------------
# Precomputed results (written by scripts/run_experiments.py)
# ---------------------------------------------------------------------------


def has_results() -> bool:
    """True if the experiment outputs are present."""
    return (RESULTS / "summary.json").exists()


def require_results() -> None:
    """Stop the page with a hint when the results have not been made."""
    if not has_results():
        st.warning("No results found. Run `python scripts/"
                   "run_experiments.py` from the project folder first.")
        st.stop()


@st.cache_data
def table(name: str) -> pd.DataFrame:
    """Load one results table from ``data/results``."""
    return pd.read_csv(RESULTS / f"{name}.csv")


@st.cache_data
def summary() -> dict:
    """Headline numbers of the study."""
    return json.loads((RESULTS / "summary.json").read_text())


@st.cache_data
def manifest() -> dict:
    """Environment and timing information of the experiment run."""
    return json.loads((RESULTS / "manifest.json").read_text())


@st.cache_data
def maps() -> Dict[str, np.ndarray]:
    """Downsampled maps of the study maze (uint8, 255 = no data)."""
    with np.load(RESULTS / "maps.npz") as data:
        return {key: data[key] for key in data.files}


def decode(values: np.ndarray) -> np.ndarray:
    """uint8 map -> floats in [0, 1], NaN where there is no data."""
    out = values.astype(float) / 254
    out[values == 255] = np.nan
    return out


def colorize(values: np.ndarray, zoom: int = 2,
             background: Sequence[int] = (246, 247, 251)) -> np.ndarray:
    """Colour a [0, 1] map with viridis; NaN cells get ``background``."""
    image = np.empty(values.shape + (3,), dtype=np.uint8)
    image[:] = background
    seen = ~np.isnan(values)
    image[seen] = gradient(values[seen])
    return image.repeat(zoom, axis=0).repeat(zoom, axis=1)


def break_even(build_s: float, slow_s: float, fast_s: float) -> float:
    """Queries needed before preprocessing pays for itself (inf if never)."""
    saving = slow_s - fast_s
    return build_s / saving if saving > 0 else float("inf")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def style(fig, height: int = 380, title: str = "", legend: bool = True):
    """Apply the app's layout defaults to a Plotly figure."""
    layout = {"height": height, "margin": {"l": 10, "r": 10,
                                           "t": 60 if title else 30,
                                           "b": 10},
              "showlegend": legend, "font": {"size": 13}}
    if title:
        layout["title"] = {"text": title, "x": 0, "xanchor": "left"}
    if legend:
        layout["legend"] = {"orientation": "h", "yanchor": "bottom",
                            "y": 1.02, "xanchor": "left", "x": 0}
    fig.update_layout(**layout)
    return fig


def chart(fig) -> None:
    """Show a Plotly figure without the Plotly logo in the toolbar."""
    st.plotly_chart(fig, config={"displaylogo": False})


# ---------------------------------------------------------------------------
# Live computations for the Maze Lab (cached per input)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Generating the maze...")
def make_maze_text(generator: str, size: int, seed: int, loops: float,
                   exit_mode: str) -> str:
    """Generate a maze and return it in the text file format."""
    maze = generate_maze(size, algorithm=generator, seed=seed)
    if loops:
        maze = braid(maze, loops, seed)
    if exit_mode != "corner":
        maze = with_endpoints(maze, goal=pick_cell(maze, exit_mode, seed))
    return maze_to_text(maze)


@st.cache_data(show_spinner="Running the searches...")
def run_searches(text: str, keys: Tuple[str, ...]
                 ) -> Tuple[dict, List[dict]]:
    """Structure statistics plus one traced run per algorithm."""
    maze = load_maze(text)
    stats = maze_stats(maze)
    runs = []
    for key in keys:
        started = time.perf_counter()
        order, result = expansion_order(maze, ALGORITHMS[key].search)
        runs.append({"key": key, "label": result.algorithm,
                     "order": as_grid(order, maze).astype(np.int32),
                     "result": result,
                     "seconds": time.perf_counter() - started})
    return stats, runs


@st.cache_data(show_spinner="Animating the race...")
def race_gif(text: str, keys: Tuple[str, ...], frames: int = 40) -> bytes:
    """Animated GIF of the selected searches racing on one clock."""
    maze = load_maze(text)
    _, runs = run_searches(text, keys)
    scale = max(1, min(6, 300 // maze.width))
    triples = [(run["label"], run["order"], run["result"]) for run in runs]
    return to_gif(race_frames(maze, triples, frames=frames, scale=scale),
                  duration_ms=80)
