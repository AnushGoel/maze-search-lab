"""Render mazes and search traces as images.

Requires NumPy and Pillow (the ``app`` extra). Images are ``uint8`` RGB
arrays that Streamlit can show directly, or that can be saved as PNG or
stitched into an animated GIF.
"""

from __future__ import annotations

import io
import warnings
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .core import WALL, Cell, Maze, SearchResult

WALL_RGB = np.array([28, 32, 44], dtype=np.uint8)
OPEN_RGB = np.array([246, 247, 251], dtype=np.uint8)
PATH_RGB = np.array([255, 56, 96], dtype=np.uint8)
START_RGB = np.array([0, 214, 143], dtype=np.uint8)
GOAL_RGB = np.array([255, 196, 0], dtype=np.uint8)
# Viridis colour stops (perceptually uniform, readable when printed in
# grey and by most readers with colour-vision deficiencies).
_STOPS = np.array([[68, 1, 84], [71, 44, 122], [59, 81, 139],
                   [44, 113, 142], [33, 144, 141], [39, 173, 129],
                   [92, 200, 99], [170, 220, 50], [253, 231, 37]],
                  dtype=float)

Run = Tuple[str, np.ndarray, SearchResult]


def gradient(values: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to viridis-like RGB colours (``uint8``)."""
    scaled = np.clip(np.asarray(values, dtype=float), 0, 1)
    scaled = scaled * (len(_STOPS) - 1)
    low = np.floor(scaled).astype(int)
    high = np.minimum(low + 1, len(_STOPS) - 1)
    frac = (scaled - low)[..., None]
    return (_STOPS[low] * (1 - frac) + _STOPS[high] * frac).astype(np.uint8)


def wall_mask(maze: Maze) -> np.ndarray:
    """Boolean (height, width) array that is True on wall cells."""
    raw = "".join(maze.rows).encode("ascii", "replace")
    grid = np.frombuffer(raw, dtype=np.uint8)
    return grid.reshape(maze.height, maze.width) == ord(WALL)


def as_grid(flat: Sequence[int], maze: Maze) -> np.ndarray:
    """Reshape a flat per-cell array (row-major) to (height, width)."""
    return np.asarray(flat, dtype=np.int64).reshape(maze.height, maze.width)


def render_maze(maze: Maze, order: Optional[np.ndarray] = None,
                path: Optional[Sequence[Cell]] = None,
                upto: Optional[int] = None, scale: int = 1) -> np.ndarray:
    """Draw a maze, optionally coloured by expansion order, with a path.

    Args:
        maze: The maze to draw.
        order: (height, width) expansion steps, -1 where never expanded.
        path: Cells to paint as the route.
        upto: Colour only cells expanded before this step (animation).
        scale: Zoom factor; each cell becomes ``scale`` x ``scale`` px.

    Returns:
        A ``uint8`` RGB array.
    """
    walls = wall_mask(maze)
    image = np.where(walls[..., None], WALL_RGB, OPEN_RGB).astype(np.uint8)
    if order is not None:
        order = np.asarray(order)
        seen = order >= 0
        if upto is not None:
            seen &= order < upto
        top = max(int(order.max()), 1)
        image[seen] = gradient(order[seen] / top)
    if path:
        rows, cols = zip(*path)
        image[list(rows), list(cols)] = PATH_RGB
    for cell, color in ((maze.start, START_RGB), (maze.goal, GOAL_RGB)):
        if 0 <= cell[0] < maze.height and 0 <= cell[1] < maze.width:
            image[cell] = color
    if scale > 1:
        image = image.repeat(scale, axis=0).repeat(scale, axis=1)
    return image


def _font(size: int) -> ImageFont.ImageFont:
    """Pillow's built-in font at ``size`` points where supported."""
    try:
        return ImageFont.load_default(size=size)
    except (TypeError, OSError, ValueError):
        return ImageFont.load_default()


def race_frames(maze: Maze, runs: Sequence[Run], frames: int = 36,
                scale: int = 3, hold: int = 8) -> List[Image.Image]:
    """Frames of a side-by-side race between several searches.

    All panels share one clock measured in expanded cells, so a panel
    that finishes early belongs to an algorithm that needed fewer
    expansions. A panel shows its route once its search has finished.

    Args:
        maze: The maze that every run searched.
        runs: ``(label, order grid, result)`` triples.
        frames: Number of animation steps before the final hold.
        scale: Zoom factor for each panel.
        hold: Extra copies of the final frame (a pause before looping).
    """
    longest = max(max(result.nodes_expanded for _, _, result in runs), 1)
    steps = np.linspace(0, longest, frames).round().astype(int).tolist()
    font = _font(13)
    label_height, gap = 22, 8
    panel_h, panel_w = maze.height * scale, maze.width * scale
    width = len(runs) * panel_w + (len(runs) - 1) * gap
    images = []
    for step in steps + [longest] * hold:
        canvas = Image.new("RGB", (width, panel_h + label_height), "white")
        draw = ImageDraw.Draw(canvas)
        for k, (label, order, result) in enumerate(runs):
            done = step >= result.nodes_expanded
            panel = render_maze(maze, order, result.path if done else None,
                                upto=step, scale=scale)
            left = k * (panel_w + gap)
            canvas.paste(Image.fromarray(panel), (left, label_height))
            shown = min(step, result.nodes_expanded)
            text = f"{label}  {shown:,}" + ("  done" if done else "")
            draw.text((left + 2, 4), text, fill=(25, 25, 35), font=font)
        images.append(canvas)
    return images


def to_png(image: np.ndarray) -> bytes:
    """Encode an RGB array as PNG bytes."""
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def to_gif(frames: Sequence[Image.Image], duration_ms: int = 70,
           colors: int = 64) -> bytes:
    """Encode frames as a looping GIF with one shared palette.

    A shared palette, taken from the last (most colourful) frame, avoids
    flicker between frames and keeps the file small.
    """
    palette = frames[-1].quantize(colors=colors,
                                  method=Image.Quantize.MEDIANCUT)
    quantized = [frame.quantize(palette=palette, dither=Image.Dither.NONE)
                 for frame in frames]
    buffer = io.BytesIO()
    quantized[0].save(buffer, format="GIF", save_all=True,
                      append_images=quantized[1:], duration=duration_ms,
                      loop=0, optimize=True)
    return buffer.getvalue()


def downsample(grid: np.ndarray, factor: int,
               reducer: Callable[..., np.ndarray] = np.nanmean
               ) -> np.ndarray:
    """Shrink a 2-D array by ``factor``, applying ``reducer`` per block.

    NaN entries are ignored by the NaN-aware reducers, so masked cells
    (walls, unexplored cells) do not distort the block values.
    """
    height, width = grid.shape
    out_h, out_w = -(-height // factor), -(-width // factor)
    padded = np.full((out_h * factor, out_w * factor), np.nan)
    padded[:height, :width] = grid
    blocks = padded.reshape(out_h, factor, out_w, factor).swapaxes(1, 2)
    blocks = blocks.reshape(out_h, out_w, factor * factor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return reducer(blocks, axis=2)
