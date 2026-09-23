"""Hand-authorable land maps, as an alternative to painting a PNG mask.

A land map is a small JSON file describing which grid cells are land, either
directly (a grid) or as axis-aligned rectangles (a vector shape list). Loading
one returns the same ``HybridEnv`` that ``load_from_image`` produces, so it
plugs into everything that already consumes one: the GUI's ``ScenarioSession``
(``.obstacles`` / ``.x_range`` / ``.y_range`` / ``.resolution``), Theta*
planning, and ``crowd_sim``'s ``set_obstacles`` / ``set_init_para`` (via
``get_static_obstacles`` / ``get_env_para``).

Grid format (explicit occupied cells, grid-index coordinates)::

    {"x_range": 20, "y_range": 20, "resolution": 1.0,
     "cells": [[3, 4], [3, 5], [4, 5]]}

Vector format (axis-aligned rectangles, grid-index coordinates, each
rectangle filling every cell it overlaps)::

    {"x_range": 20, "y_range": 20, "resolution": 1.0,
     "rects": [{"x": 5, "y": 5, "w": 4, "h": 3}]}

A map may use "cells", "rects", or both (the resulting land is their union).
Either way the outer boundary is added automatically, matching
``load_from_image``'s behaviour, so a route can never plan off the edge.
"""
import json
from pathlib import Path

from crowd_nav.utils.theta_star import HybridEnv


def _rect_cells(rect):
    x, y, w, h = rect['x'], rect['y'], rect['w'], rect['h']
    if w <= 0 or h <= 0:
        raise ValueError(f'Rectangle must have positive width/height: {rect}')
    for i in range(x, x + w):
        for j in range(y, y + h):
            yield (i, j)


def land_map_to_env(data, resolution=None, safety_distance=0):
    """Build a ``HybridEnv`` from a parsed land map dict.

    :param data: dict with ``x_range``, ``y_range``, optional ``resolution``,
        and at least one of ``cells`` / ``rects``
    :param resolution: overrides ``data['resolution']`` when given
    :param safety_distance: as ``HybridEnv``'s constructor — inflates land cells
        by this many grid steps, for building a planning grid alongside a
        raw (safety_distance=0) simulation grid, as test.py does for PNG maps
    """
    x_range, y_range = data['x_range'], data['y_range']
    res = resolution if resolution is not None else data.get('resolution', 1.0)
    if 'cells' not in data and 'rects' not in data:
        raise ValueError('Land map needs a "cells" and/or "rects" entry')

    cells = {tuple(c) for c in data.get('cells', [])}
    for rect in data.get('rects', []):
        cells.update(_rect_cells(rect))
    for (i, j) in cells:
        if not (0 <= i < x_range and 0 <= j < y_range):
            raise ValueError(f'Land cell {(i, j)} is outside the {x_range}x{y_range} grid')

    env = HybridEnv(x_range=x_range, y_range=y_range, resolution=res, safety_distance=safety_distance)
    # match load_from_image: force the outer ring so nothing plans off the edge
    for i in range(x_range):
        cells.add((i, 0))
        cells.add((i, y_range - 1))
    for j in range(y_range):
        cells.add((0, j))
        cells.add((x_range - 1, j))
    env.update(cells)
    env.x_min, env.y_min = 0, 0
    env.x_max, env.y_max = x_range * res, y_range * res
    return env


def load_land_map(path, safety_distance=0):
    """Load a land map JSON file, as ``load_static_map`` does for PNGs."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Land map not found: {path}')
    with open(path) as fh:
        data = json.load(fh)
    return land_map_to_env(data, safety_distance=safety_distance)


def load_static_map(path, x_range=50, y_range=50, resolution=1.0, safety_distance=0):
    """Load a static map (PNG land mask or land_map.py JSON) into a HybridEnv.

    Used by crowd_nav/test.py and by ``evaluate.build_grids`` (and through it the GUI's
    ``ScenarioSession``), so every entry point accepts either map format identically.

    :param path: path to a dark-pixels-are-land PNG, or a land_map.py JSON file
    :param safety_distance: as HybridEnv's constructor — pass 1 for the inflated
        planning grid test.py/evaluate.py use for Theta*, 0 for the simulator's
        actual geometry
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Static map not found: {path}')
    if path.suffix.lower() == '.json':
        return load_land_map(path, safety_distance=safety_distance)
    env = HybridEnv(x_range=x_range, y_range=y_range, resolution=resolution, safety_distance=safety_distance)
    env.load_from_image(str(path))
    return env
