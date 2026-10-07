"""Create the two small images used in the README (``docs/assets``).

Usage::

    python scripts/make_assets.py

The race GIF is rebuilt from a seeded maze; the map image is drawn from
``data/results/maps.npz``, so run the experiments first.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mazesearch import ALGORITHMS, generate_maze  # noqa: E402
from mazesearch.analysis import expansion_order  # noqa: E402
from mazesearch.render import as_grid, gradient, race_frames, to_gif  # noqa

ASSETS = ROOT / "docs" / "assets"
RESULTS = ROOT / "data" / "results"


def font(size: int) -> ImageFont.ImageFont:
    """Pillow's built-in font at ``size`` points where supported."""
    try:
        return ImageFont.load_default(size=size)
    except (TypeError, OSError, ValueError):
        return ImageFont.load_default()


def make_race() -> int:
    """Animated race of BFS, DFS and A* on a 41 x 41 maze."""
    maze = generate_maze(41, algorithm="backtracker", seed=7)
    runs = []
    for key in ("bfs", "dfs", "astar"):
        order, result = expansion_order(maze, ALGORITHMS[key].search)
        runs.append((result.algorithm, as_grid(order, maze), result))
    frames = race_frames(maze, runs, frames=40, scale=4, hold=12)
    data = to_gif(frames, duration_ms=90, colors=48)
    (ASSETS / "search_race.gif").write_bytes(data)
    return len(data)


def make_maps() -> int:
    """Where BFS, DFS, A* and landmark A* searched on the study maze."""
    with np.load(RESULTS / "maps.npz") as data:
        maps = {key: data[key] for key in data.files}
    bench = pd.read_csv(RESULTS / "benchmark.csv").set_index("algorithm")
    panels = [("order_BFS", "BFS"), ("order_DFS", "DFS"),
              ("order_A*", "A*"), ("order_ALT8", "A* + ALT (k=8)")]
    size, gap, top = 250, 12, 34
    canvas = Image.new("RGB", (4 * size + 3 * gap, size + top), "white")
    draw = ImageDraw.Draw(canvas)
    for k, (key, label) in enumerate(panels):
        raw = maps[key]
        values = raw.astype(float) / 254
        image = np.full(raw.shape + (3,), 238, dtype=np.uint8)
        image[raw != 255] = gradient(values[raw != 255])
        tile = Image.fromarray(image).resize((size, size), Image.BILINEAR)
        left = k * (size + gap)
        canvas.paste(tile, (left, top))
        count = bench.loc[label, "nodes_expanded"]
        draw.text((left, 6), f"{label}: {count:,} expanded", fill=(25, 25, 35),
                  font=font(15))
    small = canvas.convert("P", palette=Image.ADAPTIVE, colors=96)
    small.save(ASSETS / "exploration_maps.png", optimize=True)
    return (ASSETS / "exploration_maps.png").stat().st_size


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    print(f"search_race.gif: {make_race() / 1024:.0f} kB")
    print(f"exploration_maps.png: {make_maps() / 1024:.0f} kB")
