"""The scenario behind the GUI, without a line of Qt.

Everything the window does to a scenario lives here: pick a map, place the ship, plan its
route, drop crossing obstacle ships, run one episode, export it. Keeping this layer free
of Qt is what makes it testable with plain pytest and reusable from a script.

``crowd_nav.evaluate`` imports torch at module level, so it is imported lazily inside
:meth:`ScenarioSession.__init__` and :meth:`ScenarioSession.run_episode`. Importing this
module stays cheap and does not drag in the network stack.

Coordinates
-----------
The maps are 50x50 cell images. Cell ``(x, y)`` is world ``(x + 0.5, y + 0.5)``, and the
image's first row is the world's *top* (``load_from_image`` flips it). ``land_mask[y, x]``
therefore pairs with ``imshow(..., origin='lower', extent=(0, 50, 0, 50))``.
"""
import configparser
import math
import os
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from crowd_nav.utils.theta_star import ThetaStar

GRID_X, GRID_Y, GRID_RES = 50, 50, 1
MAP_X_MIN, MAP_Y_MIN = 0.0, 0.0
MAP_X_MAX, MAP_Y_MAX = GRID_X * GRID_RES, GRID_Y * GRID_RES

STATES = ('no_map', 'map', 'start_set', 'goal_set', 'planned', 'obstacles', 'ran')


class ScenarioError(Exception):
    """A problem the user can fix and retry; the GUI shows the message verbatim."""


@dataclass
class EpisodeResult:
    """One finished episode: its metrics row, its frames, and the env that produced them.

    ``context`` is the scenario as it was when the episode ran (map, seed, route,
    obstacles, settings), so an export describes this episode even if the window has
    moved on since.
    """

    metrics: dict
    states: list
    env: object
    warnings: list = field(default_factory=list)
    context: dict = field(default_factory=dict)

    def render_env(self):
        """``env`` carrying *this* result's frames.

        ``env.reset()`` rebinds ``env.states``, so an older result would otherwise replay
        whatever episode ran last.
        """
        self.env.states = self.states
        return self.env


class ScenarioSession(object):
    """State machine: no_map -> map -> start_set -> goal_set -> planned -> obstacles -> ran."""

    def __init__(self, model_dir, policy_name='sarl', safety_distance=1, colregs=False,
                 local_goal_dist=1.0, monitor_dist=5.0, monitor_exit_dist=None,
                 lookahead_rate=1.0, shore_clearance=0.5, device=None, policy=None):
        self.model_dir = model_dir
        self.policy_name = policy_name
        self.safety_distance = safety_distance
        self.colregs = colregs
        self.local_goal_dist = local_goal_dist
        self.monitor_dist = monitor_dist
        self.monitor_exit_dist = monitor_exit_dist
        self.lookahead_rate = lookahead_rate
        self.shore_clearance = shore_clearance

        env_config_path = os.path.join(model_dir, 'env.config')
        if not os.path.exists(env_config_path):
            raise ScenarioError('Model config not found: {}'.format(env_config_path))
        self.env_config = configparser.RawConfigParser()
        self.env_config.read(env_config_path)

        if policy is None:
            import torch
            from crowd_nav import evaluate
            self.device = torch.device('cpu') if device is None else torch.device(device)
            policy = evaluate.build_policy(
                SimpleNamespace(policy=policy_name, model_dir=model_dir,
                                policy_config='policy.config', il=False),
                self.device)
        else:
            self.device = device
        self.policy = policy

        self.map_path = None
        self.sim_grid = None
        self.plan_grid = None
        self.land_mask = np.zeros((GRID_Y, GRID_X), dtype=bool)
        self.margin_mask = np.zeros((GRID_Y, GRID_X), dtype=bool)
        self.env = None
        self.robot = None
        self.start_cell = None
        self.start_centre = None
        self.goal_cell = None
        self.goal_centre = None
        self.scenario = None
        self.route = None
        self.result = None
        self.warnings = []
        self.last_seed = None
        self.last_requested = None

    # ------------------------------------------------------------------ state

    @property
    def state(self):
        if self.plan_grid is None:
            return 'no_map'
        if self.start_cell is None:
            return 'map'
        if self.goal_cell is None:
            return 'start_set'
        if self.scenario is None:
            return 'goal_set'
        if self.result is not None:
            return 'ran'
        if self.scenario['dyn']:
            return 'obstacles'
        return 'planned'

    @property
    def time_step(self):
        return self.env.time_step if self.env is not None else 0.25

    @property
    def has_route(self):
        return self.route is not None

    # -------------------------------------------------------------------- map

    @staticmethod
    def world_from_click(x, y):
        """Click -> world position, or None when it is None or off the map."""
        if x is None or y is None:
            return None
        if not (MAP_X_MIN <= x <= MAP_X_MAX and MAP_Y_MIN <= y <= MAP_Y_MAX):
            return None
        return (float(x), float(y))

    @staticmethod
    def _mask(grid):
        mask = np.zeros((grid.y_range, grid.x_range), dtype=bool)
        for x, y in grid.obstacles:
            mask[y, x] = True
        return mask

    def load_map(self, png_path):
        """Load a PNG map, rebuild both grids and the simulator; clears any scenario."""
        from crowd_nav import evaluate
        png_path = str(png_path)
        if not os.path.exists(png_path):
            raise ScenarioError('Map file not found: {}'.format(png_path))
        try:
            sim_grid, plan_grid = evaluate.build_grids(png_path, self.safety_distance)
        except ScenarioError:
            raise
        except Exception as exc:      # unreadable / corrupt image
            raise ScenarioError('Cannot read map {}: {}'.format(png_path, exc))
        self.sim_grid, self.plan_grid = sim_grid, plan_grid
        self.land_mask = self._mask(sim_grid)
        self.margin_mask = self._mask(plan_grid) & ~self.land_mask
        self.env, self.robot = evaluate.build_env(
            self.env_config, SimpleNamespace(colregs=self.colregs), self.policy, sim_grid)
        self.map_path = png_path
        self.clear_scenario()
        return self.env

    def clear_scenario(self):
        self.start_cell = self.start_centre = None
        self.goal_cell = self.goal_centre = None
        self.invalidate()

    def invalidate(self):
        """Drop the route, the obstacles and the result, but keep the two chosen points."""
        self.scenario = None
        self.route = None
        self.result = None
        self.warnings = []
        self.last_seed = None

    # ------------------------------------------------------- ship start / goal

    def snap(self, pos):
        """World position -> (cell, cell centre), the position the ship actually starts at."""
        cell = self.sim_grid.world_to_grid(pos[0], pos[1])
        return cell, self.sim_grid.grid_to_world(*cell)

    def _accept_point(self, pos):
        if pos is None:
            raise ScenarioError('That point is off the canvas')
        if self.plan_grid is None:
            raise ScenarioError('Load a map first')
        cell, centre = self.snap(pos)
        for point in (pos, centre):
            if self.plan_grid.is_on_land(point):
                raise ScenarioError('({:.1f}, {:.1f}) is land or inside the safety belt - click open water'
                                    .format(pos[0], pos[1]))
        return cell, centre

    def set_start(self, pos):
        cell, centre = self._accept_point(pos)
        self.start_cell, self.start_centre = cell, centre
        self.invalidate()
        return cell, centre

    def set_goal(self, pos):
        cell, centre = self._accept_point(pos)
        self.goal_cell, self.goal_centre = cell, centre
        self.invalidate()
        return cell, centre

    # ------------------------------------------------------------------ route

    def plan_route(self):
        """Theta* on the inflated grid; stores the goal-first path and a start-first route."""
        if self.plan_grid is None:
            raise ScenarioError('Load a map first')
        if self.start_cell is None or self.goal_cell is None:
            raise ScenarioError('Set a start and a goal first')
        if self.start_cell == self.goal_cell:
            raise ScenarioError('Start and goal are in the same cell')
        cost, path, _ = ThetaStar(start=self.start_cell, goal=self.goal_cell,
                                  env=self.plan_grid).plan()
        if not path:
            raise ScenarioError('No route between the start and the goal')
        self.scenario = {'start': self.start_cell, 'goal': self.goal_cell,
                         'cost': cost, 'path': path, 'dyn': []}
        # set_local_goal_list wants the goal first; the obstacle generator wants the ship's
        # own route start first, so keep that one around separately.
        self.route = self.sim_grid.grid_2_drl(path)[::-1]
        self.result = None
        self.warnings = []
        return self.scenario

    # -------------------------------------------------------------- obstacles

    def generate_obstacles(self, n, seed, min_dist=3.0, cross_dist=(6.0, 14.0), max_trials=400):
        """Place n crossing obstacle ships along the route, reproducibly from `seed`.

        `min_dist` is the spacing between ship end points, `cross_dist` the (min, max)
        distance from the route at which a ship spawns, and `max_trials` the attempts
        per ship before it is skipped.
        """
        if self.scenario is None or self.route is None:
            raise ScenarioError('Plan a route first')
        if cross_dist[0] > cross_dist[1]:
            raise ScenarioError('Spawn distance min {} is above max {}'.format(*cross_dist))
        if n <= 0:
            self.scenario['dyn'] = []
            self.result = None
            self.warnings = []
            self.last_seed = seed
            self.last_requested = 0
            return []
        obstacles = self.plan_grid.generate_crossing_obstacles(
            n, self.route, forb_list=[self.start_centre, self.goal_centre],
            min_dist=min_dist, cross_dist=tuple(cross_dist), max_trials=max_trials,
            rng=np.random.RandomState(seed))
        if not obstacles:
            raise ScenarioError('Seed {} could not place any obstacle ship - try another seed or a smaller count'.format(seed))
        self.warnings = []
        if len(obstacles) < n:
            self.warnings.append('Placed {} of {} obstacle ships'.format(len(obstacles), n))
        self.scenario['dyn'] = obstacles
        self.result = None
        self.last_seed = seed
        self.last_requested = n
        return obstacles

    # ---------------------------------------------------------------- episode

    def run_episode(self, max_steps=2000, on_step=None):
        """Run one episode. ``on_step(k, n)`` may raise EpisodeCancelled to abort."""
        from crowd_nav import evaluate
        if self.env is None:
            raise ScenarioError('Load a map first')
        if self.scenario is None:
            raise ScenarioError('Plan a route first')
        cfg = SimpleNamespace(max_steps=max_steps, no_global_plan=False,
                              local_goal_dist=self.local_goal_dist,
                              monitor_dist=self.monitor_dist,
                              monitor_exit_dist=self.monitor_exit_dist,
                              lookahead_rate=self.lookahead_rate,
                              shore_clearance=self.shore_clearance)
        context = self._context(max_steps)   # before the run: it reads the scenario as set
        row = evaluate.run_episode(self.env, self.robot, self.sim_grid, self.scenario, cfg,
                                   on_step=on_step)
        self.result = EpisodeResult(metrics=row, states=self.env.states, env=self.env,
                                    warnings=list(self.warnings), context=context)
        return self.result

    def _context(self, max_steps):
        """The scenario and settings of the episode about to run, for its exports."""
        dyn = self.scenario['dyn']
        return {
            'map': os.path.splitext(os.path.basename(self.map_path))[0],
            'map_path': self.map_path,
            'model_dir': self.model_dir,
            'policy': getattr(self.policy, 'name', self.policy_name),
            'colregs': self.colregs,
            'seed': self.last_seed if dyn else '',
            'requested_ships': self.last_requested if dyn else 0,
            'max_steps': max_steps,
            'time_step': self.time_step,
            'start': list(self.start_centre),
            'goal': list(self.goal_centre),
            'route': [list(p) for p in self.route],
            'obstacles': [dict(zip(('px', 'py', 'gx', 'gy'), row)) for row in self.obstacle_table()],
            'local_goal_dist': self.local_goal_dist,
            'monitor_dist': self.monitor_dist,
            'shore_clearance': self.shore_clearance,
        }

    # ----------------------------------------------------------------- export

    def export_video(self, result, path, fps=8, progress=None):
        """Write the episode to mp4 (gif without ffmpeg) and return the path written.

        Uses its own offscreen Agg figure, so it never touches the window's canvas.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from crowd_sim.envs.render import EpisodeAnimation, save_animation

        figure = Figure(figsize=(7, 7))
        FigureCanvasAgg(figure)
        ax = figure.add_subplot(111)
        scene = EpisodeAnimation(result.render_env(), ax)
        anim = scene.animate(figure, result.env.time_step * 1000)
        return save_animation(anim, path, fps=fps, progress=progress)

    # Columns a results CSV adds after crowd_nav.evaluate's, which describe the batch runs.
    CSV_EXTRA_FIELDS = ['policy', 'colregs', 'model_dir', 'map_path']

    def export_results(self, result, path):
        """Write the episode's results to ``path`` and return the path written.

        ``.csv`` is one row with the columns of ``crowd_nav.evaluate``'s result CSVs
        (then policy, colregs, model and map path), so a GUI run lines up with a batch
        run. ``.json`` is the full record: settings, metrics, route, obstacle ships and
        every ship's position and velocity at every step.
        """
        import csv
        import json
        from crowd_nav import evaluate
        path = str(path)
        kind = os.path.splitext(path)[1].lower()
        if kind not in ('.csv', '.json'):
            raise ScenarioError('Choose a .csv or .json file, not {}'.format(
                os.path.basename(path)))
        context = result.context
        try:
            if kind == '.csv':
                row = dict.fromkeys(evaluate.CSV_FIELDS, '')
                row.update(result.metrics)
                row.update(map=context.get('map', ''), seed=context.get('seed', ''),
                           requested_n_dynamic=context.get('requested_ships', ''))
                row.update({key: context.get(key, '') for key in self.CSV_EXTRA_FIELDS})
                with open(path, 'w', newline='') as fh:
                    writer = csv.DictWriter(fh, fieldnames=evaluate.CSV_FIELDS + self.CSV_EXTRA_FIELDS,
                                            extrasaction='ignore')
                    writer.writeheader()
                    writer.writerow(row)
            else:
                record = {'settings': {k: v for k, v in context.items()
                                       if k not in ('route', 'obstacles')},
                          'metrics': result.metrics,
                          'warnings': result.warnings,
                          'route': context.get('route', []),
                          'obstacles': context.get('obstacles', []),
                          'trajectory': self._trajectory(result)}
                with open(path, 'w') as fh:
                    json.dump(record, fh, indent=1, default=float)
        except OSError as exc:
            raise ScenarioError('Cannot write {}: {}'.format(path, exc.strerror or exc))
        return path

    @staticmethod
    def _trajectory(result):
        """Per step: time, own ship (x, y, vx, vy) and each obstacle ship (x, y, vx, vy)."""
        dt = result.context.get('time_step', result.env.time_step)

        def motion(s):
            return [round(float(v), 4) for v in (s.px, s.py, s.vx, s.vy)]

        return [{'t': round(k * dt, 4), 'own_ship': motion(own), 'ships': [motion(s) for s in ships]}
                for k, (own, ships) in enumerate(result.states)]

    # ------------------------------------------------------------------ misc

    def route_length(self):
        """Length of the planned route in metres, or None before a route exists."""
        if self.route is None:
            return None
        return sum(math.dist(a, b) for a, b in zip(self.route, self.route[1:]))

    def obstacle_table(self):
        """The placed obstacles as (px, py, gx, gy) rows, for the GUI table."""
        if self.scenario is None:
            return []
        return [(o['px'], o['py'], o['gx'], o['gy']) for o in self.scenario['dyn']]
