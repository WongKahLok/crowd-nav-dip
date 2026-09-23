"""Frozen benchmark scenarios P5, P8, P12 and P15, saved as JSON next to the maps.

A scenario file fixes everything random about one test difficulty: for every map it
covers, the ship's start and goal, its Theta* route, and each obstacle ship with its own
land-free path. ``test.py --scenario``, ``evaluate.py --scenario`` and the GUI's scenario
dropdown all read the same file, so they run the identical episode.

File layout (``static_maps/scenarios/P8.json``)::

    {"format": 1, "name": "P8", "n_obstacles": 8,
     "maps": {"land_shapes_sf_crop": {
         "map": "static_maps/land_shapes_sf_crop.png", "map_sha256": "...",
         "seed": 3, "start": [40, 5], "goal": [32, 42], "cost": 41.2,
         "path": [[32, 42], ..., [40, 5]], "placed": 8,
         "obstacles": [{"px": .., "py": .., "gx": .., "gy": .., "path": [[x, y], ...]}]}}}

``start``, ``goal`` and ``path`` are grid cells (path goal first, as Theta* returns it);
obstacle coordinates are world metres. ``map`` is relative to the repository root, and
``map_sha256`` makes a scenario refuse to run on a map that changed after the freeze.

The obstacle ships come from :func:`place_obstacles`, which the GUI's *Generate random*
also uses: the same map, route, count and seed give the same ships in both places.

Regenerate the files (only when a map or the generator changes on purpose)::

    python -m crowd_nav.utils.scenario
"""
import argparse
import hashlib
import json
import os

import numpy as np

from crowd_nav.utils.land_map import load_static_map
from crowd_nav.utils.theta_star import ThetaStar

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENARIO_DIR = os.path.join(REPO_ROOT, 'static_maps', 'scenarios')
FORMAT = 1

PHASES = {'P5': 5, 'P8': 8, 'P12': 12, 'P15': 15}
DEFAULT_MAPS = ('static_maps/land_shapes_sf_crop.png', 'static_maps/land_shapes_ht_crop.png')
# the ship's task in test.py and in the recorded demo videos
DEFAULT_START, DEFAULT_GOAL = (40, 5), (32, 42)
SAFETY_DISTANCE = 1      # planning belt, as test.py, evaluate.py and the GUI use
MAX_SEED_TRIES = 200
REFILL_ROUNDS = 3        # extra placement passes for ships the first pass could not fit


class ScenarioFileError(ValueError):
    """A scenario file that is missing, malformed, or does not match the map."""


def map_key(map_path):
    """Scenario entries are keyed by the map's file name without extension."""
    return os.path.splitext(os.path.basename(str(map_path)))[0]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b''):
            digest.update(chunk)
    return digest.hexdigest()


def place_obstacles(plan_grid, route, start_centre, goal_centre, n, seed):
    """Crossing obstacle ships for one route, reproducible from ``seed``.

    Shared by the GUI session and the scenario generator, so both place the same ships.
    ``route`` is the ship's path in world coordinates, start first.

    The generator anchors ship k at a fixed fraction of the route, so an anchor in a
    narrow channel fails for every seed. Ships that did not fit are retried in up to
    REFILL_ROUNDS more passes, whose anchors fall at different points of the route,
    and which stay clear of the ships already placed.
    """
    if n <= 0:
        return []
    rng = np.random.RandomState(seed)
    forb = [start_centre, goal_centre]
    obstacles = plan_grid.generate_crossing_obstacles(n, route, forb_list=forb, rng=rng)
    for _ in range(REFILL_ROUNDS):
        if len(obstacles) >= n:
            break
        extra = plan_grid.generate_crossing_obstacles(n - len(obstacles), route,
                                                      forb_list=forb + obstacles, rng=rng)
        if not extra:
            break
        obstacles += extra
    return obstacles


# ------------------------------------------------------------------ building

def _plain(value):
    """numpy scalars and tuples -> JSON-safe floats and lists."""
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return value


def build_entry(map_path, n, start=DEFAULT_START, goal=DEFAULT_GOAL, base_seed=0,
                safety_distance=SAFETY_DISTANCE):
    """One map's entry of a scenario: the first seed from ``base_seed`` that places all n ships.

    When no seed in MAX_SEED_TRIES does (the map has too little water along the route),
    the seed placing the most ships is kept and the entry's ``placed`` is below n.
    """
    sim_grid = load_static_map(map_path, safety_distance=0)
    plan_grid = load_static_map(map_path, safety_distance=safety_distance)
    cost, path, _ = ThetaStar(start=tuple(start), goal=tuple(goal), env=plan_grid).plan()
    if not path:
        raise ScenarioFileError('No route from {} to {} on {}'.format(start, goal, map_path))
    route = sim_grid.grid_2_drl(path)[::-1]
    start_centre = sim_grid.grid_to_world(*start)
    goal_centre = sim_grid.grid_to_world(*goal)
    best_seed, obstacles = None, []
    for seed in range(base_seed, base_seed + MAX_SEED_TRIES):
        placed = place_obstacles(plan_grid, route, start_centre, goal_centre, n, seed)
        if len(placed) > len(obstacles) or best_seed is None:
            best_seed, obstacles = seed, placed
        if len(placed) == n:
            break
    seed = best_seed
    return {
        'map': os.path.relpath(os.path.abspath(map_path), REPO_ROOT).replace(os.sep, '/'),
        'map_sha256': file_sha256(map_path),
        'seed': seed,
        'start': _plain(start), 'goal': _plain(goal),
        'cost': float(cost),
        'path': _plain(path),
        'placed': len(obstacles),
        'obstacles': [{'px': float(o['px']), 'py': float(o['py']),
                       'gx': float(o['gx']), 'gy': float(o['gy']),
                       'path': _plain(o['path'])} for o in obstacles],
    }


def build_scenario(name, n, map_paths=DEFAULT_MAPS, **kwargs):
    maps = {}
    for map_path in map_paths:
        maps[map_key(map_path)] = build_entry(os.path.join(REPO_ROOT, map_path), n, **kwargs)
    return {'format': FORMAT, 'name': name, 'n_obstacles': n, 'maps': maps}


def save_scenario(scenario, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(scenario, fh, indent=1)
        fh.write('\n')


# ------------------------------------------------------------------- loading

def available():
    """Names of the scenario files in SCENARIO_DIR, in difficulty order where known."""
    if not os.path.isdir(SCENARIO_DIR):
        return []
    names = [os.path.splitext(f)[0] for f in os.listdir(SCENARIO_DIR) if f.endswith('.json')]
    return sorted(names, key=lambda s: (PHASES.get(s, float('inf')), s))


def resolve(name_or_path):
    """A path to an existing file, or a name like ``P8`` looked up in SCENARIO_DIR."""
    if os.path.isfile(name_or_path):
        return name_or_path
    candidate = os.path.join(SCENARIO_DIR, '{}.json'.format(name_or_path))
    if os.path.isfile(candidate):
        return candidate
    raise ScenarioFileError('Unknown scenario {!r}; available: {}'.format(
        name_or_path, ', '.join(available()) or 'none (run python -m crowd_nav.utils.scenario)'))


def load_scenario(name_or_path):
    path = resolve(name_or_path)
    with open(path, encoding='utf-8') as fh:
        data = json.load(fh)
    if data.get('format') != FORMAT or 'maps' not in data:
        raise ScenarioFileError('{} is not a format {} scenario file'.format(path, FORMAT))
    data.setdefault('name', os.path.splitext(os.path.basename(path))[0])
    return data


def map_path_of(entry):
    """Absolute path of the map an entry was built on."""
    return os.path.join(REPO_ROOT, entry['map'])


def entry_for(scenario, map_path, check_hash=True):
    """The entry of ``scenario`` for ``map_path``; refuses a map changed since the freeze."""
    key = map_key(map_path)
    if key not in scenario['maps']:
        raise ScenarioFileError('Scenario {} has no entry for map {}; it covers: {}'.format(
            scenario['name'], key, ', '.join(scenario['maps'])))
    entry = scenario['maps'][key]
    if check_hash and file_sha256(map_path) != entry['map_sha256']:
        raise ScenarioFileError('Map {} changed since scenario {} was frozen; regenerate the '
                                'scenarios or restore the map'.format(map_path, scenario['name']))
    return entry


def to_episode(entry):
    """The dict ``evaluate.run_episode`` takes, with fresh copies of every list."""
    return {
        'start': tuple(entry['start']),
        'goal': tuple(entry['goal']),
        'cost': entry['cost'],
        'path': [tuple(p) for p in entry['path']],
        'dyn': [dict(px=o['px'], py=o['py'], gx=o['gx'], gy=o['gy'],
                     path=[tuple(p) for p in o['path']]) for o in entry['obstacles']],
    }


def main(argv=None):
    parser = argparse.ArgumentParser('Regenerate the frozen P5/P8/P12/P15 scenario files')
    parser.add_argument('--out_dir', default=SCENARIO_DIR)
    parser.add_argument('--maps', nargs='+', default=list(DEFAULT_MAPS),
                        help='maps relative to the repository root')
    args = parser.parse_args(argv)
    for name, n in PHASES.items():
        scenario = build_scenario(name, n, map_paths=args.maps)
        path = os.path.join(args.out_dir, '{}.json'.format(name))
        save_scenario(scenario, path)
        print('{}: {}'.format(path, ', '.join(
            '{} seed {} placed {}/{}'.format(k, e['seed'], e['placed'], n)
            for k, e in scenario['maps'].items())))


if __name__ == '__main__':
    main()
