"""Load the static land maps used by train.py / test.py / evaluate.py.

Reuses HybridEnv.load_from_image so the GUI reads the exact same land mask
the simulation and evaluation pipeline plan and collide against, instead of
just displaying the source PNG.
"""

from pathlib import Path

from crowd_nav.utils.theta_star import HybridEnv

# same discretisation test.py and evaluate.py use (crowd_nav/evaluate.py: GRID_X/Y/RES)
GRID_X, GRID_Y, GRID_RES = 50, 50, 1

BUNDLED_MAPS_DIR = Path(__file__).resolve().parents[2] / "static_maps"


def list_bundled_maps(maps_dir=BUNDLED_MAPS_DIR):
    """PNGs shipped under static_maps/ (the maps evaluate.py's --static_map points at)."""
    maps_dir = Path(maps_dir)
    if not maps_dir.is_dir():
        return []
    return sorted(maps_dir.glob("*.png"))


def load_static_map(path, x_range=GRID_X, y_range=GRID_Y, resolution=GRID_RES):
    """Load a static map PNG into a HybridEnv, as test.py/evaluate.py do.

    :param path: path to a land-mask PNG (dark pixels are land, per HybridEnv's
        default threshold)
    :return: a HybridEnv whose `.obstacles` is the thresholded land grid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Static map not found: {path}")
    env = HybridEnv(x_range=x_range, y_range=y_range, resolution=resolution, safety_distance=0)
    env.load_from_image(str(path))
    return env
