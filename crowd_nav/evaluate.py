"""Batch evaluation of Theta* guidance and local avoidance (metrics version 2).

Normally driven through crowd_nav/exp.py, which reads the experiments table:
    python -m crowd_nav.exp run colregs

Directly, from the repository root:
    python -m crowd_nav.evaluate --policy sarl --model_dir crowd_nav/data/output-trained \
        --static_map static_maps/land_shapes_sf_crop.png --episodes 100 --densities 2,4,8

Clearances use swept circles against moving circles and filled land rectangles.
They include initial overlap and every executed motion segment. Danger fraction
includes collision steps. Simulated time comes from the episode clock; max_steps
sets its budget. Version 1 CSV metrics are not directly comparable to version 2.
"""
import argparse
import configparser
import csv
import logging
import math
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import gym
import numpy as np
import torch

from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.policy.sarl_colregs import THRESHOLDS
from crowd_nav.utils.local_goal import LocalGoalTracker
from crowd_nav.utils.theta_star import HybridEnv, ThetaStar, interpolate_path
from crowd_sim.envs.utils.info import Collision, ReachGoal, Timeout
from crowd_sim.envs.utils.robot import Robot

# this file is <repo>/crowd_nav/evaluate.py, so the repository root is two levels up
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPS = {'sf': os.path.join(REPO_ROOT, 'static_maps', 'land_shapes_sf_crop.png'),
        'ht': os.path.join(REPO_ROOT, 'static_maps', 'land_shapes_ht_crop.png'),
        'sg': os.path.join(REPO_ROOT, 'static_maps', 'land_shapes_sg_crop.png')}


def map_key(map_name):
    """The short name of a map: 'land_shapes_ht_crop', as the CSVs spell it, is 'ht'."""
    for key, path in MAPS.items():
        if os.path.splitext(os.path.basename(path))[0] == map_name:
            return key
    return map_name


GRID_X, GRID_Y, GRID_RES = 50, 50, 1  # same discretisation test.py uses

CSV_FIELDS = ['map', 'seed', 'n_dynamic', 'outcome', 'steps', 'sim_time', 'travelled_dist',
              'planned_dist', 'detour_ratio', 'min_sep_dynamic', 'min_sep_static',
              'danger_frac', 'plan_cost', 'start_x', 'start_y', 'goal_x', 'goal_y',
              'requested_n_dynamic', 'collision_kind', 'metrics_version', 'emergency_frac']


class EpisodeCancelled(Exception):
    """Raised out of an ``on_step`` callback to abandon the running episode."""


class SettingsError(Exception):
    """A setting the user can fix: a missing model folder, a misspelt override."""


# -- settings -----------------------------------------------------------------------

def add_run_settings(parser):
    """Every setting that changes how a trip unfolds, declared once, so that running,
    filming and rule-checking an experiment all use the same values."""
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy', type=str, default='sarl')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--colregs', default=False, action='store_true',
                        help='apply the COLREGs filter: --policy sarl becomes sarl_colregs')
    parser.add_argument('--no_global_plan', default=False, action='store_true',
                        help='ignore the Theta* route and steer at the final goal')
    parser.add_argument('--local_goal_dist', type=float, default=1.0)
    parser.add_argument('--monitor_dist', type=float, default=5.0)
    parser.add_argument('--monitor_exit_dist', type=float, default=None,
                        help='obstacles stop being monitored beyond this, default 1.3 * monitor_dist')
    parser.add_argument('--lookahead_rate', type=float, default=1.0, help='m/s the look-ahead may change')
    parser.add_argument('--shore_clearance', type=float, default=0.5,
                        help='room between the ship-to-goal segment and the land, on top of the radius')
    parser.add_argument('--nearby_dist', type=float, default=None,
                        help='land sensing radius in m: how far away land cells are still given to '
                             'the policy as agents (default: the map value of 5). 0 keeps land out '
                             'of the observation entirely; collision checking is unaffected')
    parser.add_argument('--max_steps', type=int, default=500)  # 500 * 0.25s = 125s per episode
    parser.add_argument('--safety_distance', type=int, default=1)
    parser.add_argument('--min_route_len', type=float, default=20.0)
    parser.add_argument('--env_set', nargs='+', default=[], metavar='SECTION.KEY=VALUE',
                        help='change env.config values for this run only, e.g. robot.visible=true '
                             '(the other ships see ours and move out of its way)')
    parser.add_argument('--policy_set', nargs='+', default=[], metavar='SECTION.KEY=VALUE',
                        help='change policy.config values for this run only, e.g. '
                             'motion.max_turn_rate=20 or colregs.enforce_stand_on=true')
    return parser


class _SettingsParser(argparse.ArgumentParser):
    def error(self, message):
        raise SettingsError('Cannot read the run settings: ' + message)


def run_settings(policy, model_dir, extra=()):
    """The settings of one experiment: its policy, its model folder and its extra flags."""
    parser = add_run_settings(_SettingsParser(add_help=False))
    return parser.parse_args(['--policy', policy, '--model_dir', model_dir] + list(extra))


# Sections a config may leave out because the code falls back to defaults, with the keys
# that are read from them. An override may add these; anything else has to exist already.
OPTIONAL_KEYS = {'colregs': tuple(THRESHOLDS) + ('enforce_stand_on',)}


def apply_overrides(config, overrides, source):
    """Apply ``section.key=value`` overrides to a loaded config.

    The key has to exist in the file or be one of ``OPTIONAL_KEYS``, so a typo is an
    error and not a quiet run of the default experiment.
    """
    for item in overrides:
        name, sep, value = item.partition('=')
        section, dot, key = name.partition('.')
        if not sep or not dot or not section or not key:
            raise SettingsError('Cannot read the override "{}": write it as section.key=value'
                                .format(item))
        if not config.has_option(section, key):
            if key not in OPTIONAL_KEYS.get(section, ()):
                raise SettingsError('"{}.{}" is not a setting of {}'.format(section, key, source))
            if not config.has_section(section):
                config.add_section(section)
        config.set(section, key, value.strip())
    return config


def read_config(cfg, which):
    """The env or policy config of a run: the model folder's own copy, then the overrides."""
    path = getattr(cfg, which + '_config', 'configs/{}.config'.format(which))
    model_dir = getattr(cfg, 'model_dir', None)
    if model_dir is not None:
        path = os.path.join(model_dir, os.path.basename(path))
        if not os.path.exists(path):
            raise SettingsError('{} not found. Is the model folder right?'.format(path))
    config = configparser.RawConfigParser()
    config.read(path)
    return apply_overrides(config, getattr(cfg, which + '_set', None) or (), path)


def policy_key(cfg):
    """The policy to build. ``--colregs`` selects the rule-following variant of SARL."""
    if getattr(cfg, 'colregs', False):
        if cfg.policy == 'sarl':
            return 'sarl_colregs'
        if cfg.policy != 'sarl_colregs':
            logging.warning('--colregs has no effect on the %s policy: only SARL has a '
                            'COLREGs filter', cfg.policy)
    return cfg.policy


def build_grids(map_path, safety_distance):
    """Two grids from one image: raw for simulated geometry, inflated for planning (as test.py)."""
    sim_grid = HybridEnv(x_range=GRID_X, y_range=GRID_Y, resolution=GRID_RES, safety_distance=0)
    sim_grid.load_from_image(map_path)
    plan_grid = HybridEnv(x_range=GRID_X, y_range=GRID_Y, resolution=GRID_RES,
                          safety_distance=safety_distance)
    plan_grid.load_from_image(map_path)
    return sim_grid, plan_grid


def free_cells(grid):
    """Every cell that is not land (obstacles here are already inflated by Grid.update)."""
    return [(x, y) for x in range(grid.x_range) for y in range(grid.y_range)
            if (x, y) not in grid.obstacles]


def sample_scenario(plan_grid, cells, n_dynamic, min_route_len, max_tries=200):
    """Random start/goal with a valid Theta* route, plus n random moving obstacles.

    Returns None if no valid scenario was found within max_tries.
    """
    if not cells:
        return None
    for _ in range(max_tries):
        start = cells[np.random.randint(len(cells))]
        goal = cells[np.random.randint(len(cells))]
        if math.dist(start, goal) < min_route_len:  # reject trivially short tasks
            continue
        cost, path, _ = ThetaStar(start=start, goal=goal, env=plan_grid).plan()
        if not path:  # plan() returns [], [], [] when the goal is unreachable
            continue
        half = GRID_RES / 2  # cell index -> cell centre, matching grid_2_drl
        forb = [(start[0] + half, start[1] + half), (goal[0] + half, goal[1] + half)]
        dyn = plan_grid.generate_dynamic_obstacles(n_dynamic, forb_list=forb, min_dist=3.0)
        return {'start': start, 'goal': goal, 'cost': cost, 'path': path, 'dyn': dyn}
    return None


def build_env(env_config, args, policy, sim_grid):
    """One simulator wired to one loaded map and one loaded policy.

    ``args`` only has to carry ``colregs``, which ``CrowdSim_test.configure`` reads.
    """
    env = gym.make('CrowdSimTest-v0')
    env.configure(env_config, args)
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)
    env.set_obstacles(sim_grid.get_static_obstacles())
    env.set_init_para(sim_grid.get_env_para())
    # Optional override of the land sensing radius (--nearby_dist); the map supplies 5 m.
    # getattr keeps callers that build their own args object working unchanged.
    nearby_dist = getattr(args, 'nearby_dist', None)
    if nearby_dist is not None:
        env.nearby_dist = nearby_dist
    policy.set_env(env)
    return env, robot


def run_episode(env, robot, sim_grid, scenario, cfg, on_step=None):
    """Run one episode; all safety metrics come from the executed transitions.

    ``on_step(steps_done, max_steps)`` is called after every executed step; raising
    :class:`EpisodeCancelled` from it abandons the episode.
    """
    if cfg.max_steps <= 0:
        raise ValueError('max_steps must be positive')
    env.time_limit = cfg.max_steps * env.time_step
    env.set_local_goal_list(sim_grid.grid_2_drl(scenario['path']))
    ob = env.reset('test', None, scenario['dyn'])
    full_path = env.global_path
    path_dense = interpolate_path(full_path)[1:]
    planned = sum(math.dist(a, b) for a, b in zip(full_path, full_path[1:]))

    if cfg.no_global_plan:
        tracker = None
        robot.gx, robot.gy = full_path[-1]
    else:
        tracker = LocalGoalTracker.from_env(
            env, path_dense, goal_dist=cfg.local_goal_dist, monitor_dist=cfg.monitor_dist,
            monitor_exit=cfg.monitor_exit_dist, lookahead_rate=cfg.lookahead_rate,
            shore_clearance=cfg.shore_clearance)
        robot.gx, robot.gy = tracker.update(robot.get_full_state(), ob)

    nav = robot.get_full_state()
    humans = [h.get_observable_state() for h in env.humans]
    _, _, initial = env.assess_motion(nav, nav, humans, humans, end_time=env.global_time)
    min_sep_dyn, min_sep_stat = initial.min_sep_dynamic, initial.min_sep_static
    travelled, danger_steps, emergency_steps, steps = 0.0, 0, 0, 0
    outcome, collision_kind = 'timeout', None

    for steps in range(1, cfg.max_steps + 1):
        ob = env.light_reset('test', None)
        action = robot.act(ob)
        emergency_steps += bool(getattr(robot.policy, 'emergency', False))
        prev = robot.get_position()
        ob, _, done, info = env.step(action)
        travelled += math.dist(prev, robot.get_position())
        min_sep_dyn = min(min_sep_dyn, info.min_sep_dynamic)
        min_sep_stat = min(min_sep_stat, info.min_sep_static)
        if min(info.min_sep_dynamic, info.min_sep_static) < env.discomfort_dist:
            danger_steps += 1
        if on_step is not None:
            on_step(steps, cfg.max_steps)

        if done:
            if isinstance(info, Collision):
                collision_kind = info.collision_kind
                outcome = 'collision_' + collision_kind
            elif isinstance(info, ReachGoal):
                if math.dist(robot.get_position(), full_path[-1]) >= robot.radius:
                    raise AssertionError('ReachGoal must refer to the final route goal')
                outcome = 'success'
            elif isinstance(info, Timeout):
                outcome = 'timeout'
            else:
                raise ValueError('Unexpected terminal info: {}'.format(info))
            break
        if tracker is not None:
            robot.gx, robot.gy = tracker.update(robot.get_full_state(), ob)

    if not math.isclose(env.global_time, steps * env.time_step, rel_tol=0, abs_tol=1e-8):
        raise AssertionError('Episode time does not match executed steps')
    return {
        'outcome': outcome,
        'steps': steps,
        'sim_time': round(env.global_time, 6),
        'travelled_dist': round(travelled, 3),
        'planned_dist': round(planned, 3),
        'detour_ratio': round(travelled / planned, 3) if planned > 0 else '',
        'min_sep_dynamic': round(min_sep_dyn, 6) if math.isfinite(min_sep_dyn) else '',
        'min_sep_static': round(min_sep_stat, 6) if math.isfinite(min_sep_stat) else '',
        'danger_frac': round(danger_steps / steps, 6) if steps else '',
        'plan_cost': round(scenario['cost'], 3),
        'start_x': scenario['start'][0], 'start_y': scenario['start'][1],
        'goal_x': scenario['goal'][0], 'goal_y': scenario['goal'][1],
        'n_dynamic': len(env.humans),
        'collision_kind': collision_kind or '',
        'metrics_version': 2,
        'emergency_frac': (round(emergency_steps / steps, 6)
                           if steps and hasattr(robot.policy, 'emergency') else ''),
    }


def scenario_failed_row(map_name, seed, requested):
    row = dict.fromkeys(CSV_FIELDS, '')
    row.update(map=map_name, seed=seed, requested_n_dynamic=requested,
               outcome='scenario_failed', metrics_version=2)
    return row


def build_policy(args, device):
    """Load the policy and its trained weights exactly the way test.py does."""
    name = policy_key(args)
    if name not in policy_factory:
        raise SettingsError('No policy called "{}". Choose from: {}'.format(
            name, ', '.join(sorted(policy_factory))))
    policy = policy_factory[name]()
    policy.configure(read_config(args, 'policy'))
    if policy.trainable:
        if args.model_dir is None:
            raise ValueError('Trainable policy needs --model_dir')
        weights = os.path.join(args.model_dir, 'il_model.pth' if args.il else 'rl_model.pth')
        resumed = os.path.join(args.model_dir, 'resumed_rl_model.pth')
        if not args.il and os.path.exists(resumed):
            weights = resumed
        policy.get_model().load_state_dict(torch.load(weights, map_location=device))
    policy.set_phase('test')
    policy.set_device(device)
    return policy


# -- trips --------------------------------------------------------------------------

class TripRunner(object):
    """Sails trips under one set of settings.

    A trip is (map, ships, seed); the seed fixes the start, the goal and the traffic, so
    the batch run, its video and the rule check all go through :meth:`run` and get the
    same trip. The policy is loaded once, each map's simulator the first time it is used.
    """

    def __init__(self, cfg, device=None):
        self.cfg = cfg
        self.env_config = read_config(cfg, 'env')
        self.policy = build_policy(cfg, device or torch.device('cpu'))
        self.stages = {}

    def stage(self, map_path):
        """(env, robot, sim_grid, plan_grid, free cells) of one map."""
        if map_path not in self.stages:
            sim_grid, plan_grid = build_grids(map_path, self.cfg.safety_distance)
            env, robot = build_env(self.env_config, self.cfg, self.policy, sim_grid)
            self.stages[map_path] = (env, robot, sim_grid, plan_grid, free_cells(plan_grid))
        return self.stages[map_path]

    def env(self, map_path):
        """The simulator of one map; after :meth:`run` it holds that trip's frames."""
        return self.stage(map_path)[0]

    def run(self, map_path, ships, seed, on_step=None):
        """Sail one trip and return its CSV row."""
        env, robot, sim_grid, plan_grid, cells = self.stage(map_path)
        map_name = os.path.splitext(os.path.basename(map_path))[0]
        # one policy serves every map, so point it at the simulator of this one
        self.policy.set_env(env)
        np.random.seed(seed)  # scenario sampling is the only stochastic part
        scenario = sample_scenario(plan_grid, cells, ships, self.cfg.min_route_len)
        if scenario is None:
            logging.warning('%s seed %d: no valid scenario found', map_name, seed)
            return scenario_failed_row(map_name, seed, ships)
        row = run_episode(env, robot, sim_grid, scenario, self.cfg, on_step=on_step)
        row.update(map=map_name, seed=seed, requested_n_dynamic=ships)
        if row['n_dynamic'] != ships:
            logging.warning('%s seed %d: placed %d of %d requested obstacles',
                            map_name, seed, row['n_dynamic'], ships)
        return row


def sail(runner, trip):
    """The plain task of :func:`run_trips`: sail the trip, return its row."""
    return runner.run(*trip)


_RUNNER = None  # the TripRunner of this worker process, see _start_worker


def _start_worker(cfg):
    """Runs once in every worker process: one policy and one set of simulators each."""
    global _RUNNER
    torch.set_num_threads(1)  # one core per trip, and replays stay bit-identical
    # The scenario generator reports every map that cannot fit all its ships, on stdout
    # and as a warning. In a batch that is one line per trip; the row records it instead.
    sys.stdout = open(os.devnull, 'w')
    logging.disable(logging.WARNING)
    _RUNNER = TripRunner(cfg)


def _work(job):
    task, trip = job
    return task(_RUNNER, trip)


def run_trips(cfg, trips, jobs=None, task=sail, on_result=None):
    """Sail many trips over ``jobs`` processes; results come back in the order of ``trips``.

    ``task(runner, trip)`` is what a worker does with each trip (a module level function,
    so it can be sent to the workers). ``on_result(done, total, result)`` is called here
    as each trip finishes. Trips are independent, so ``jobs`` changes the speed and not
    the numbers.
    """
    trips = list(trips)
    jobs = max(1, min(jobs or os.cpu_count() or 1, len(trips)))
    results = [None] * len(trips)
    if jobs == 1:
        torch.set_num_threads(1)
        runner = TripRunner(cfg)
        for i, trip in enumerate(trips):
            results[i] = task(runner, trip)
            if on_result is not None:
                on_result(i + 1, len(trips), results[i])
        return results

    os.environ.setdefault('OMP_NUM_THREADS', '1')  # inherited by the workers
    # spawn, not fork: forking a process in which torch already runs threads can deadlock
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=jobs, mp_context=context,
                             initializer=_start_worker, initargs=(cfg,)) as pool:
        futures = {pool.submit(_work, (task, trip)): i for i, trip in enumerate(trips)}
        try:
            for done, future in enumerate(as_completed(futures), 1):
                results[futures[future]] = future.result()
                if on_result is not None:
                    on_result(done, len(trips), results[futures[future]])
        except BaseException:
            # Ctrl-C or a failed trip: drop the rest, and stop the trips under way now
            # instead of sailing each of them to its end first.
            for future in futures:
                future.cancel()
            for process in list(getattr(pool, '_processes', {}).values()):
                process.terminate()
            raise
    return results


def main():
    parser = argparse.ArgumentParser('Batch evaluation over randomised scenarios')
    add_run_settings(parser)
    parser.add_argument('--static_map', type=str, nargs='+', required=True)
    parser.add_argument('--episodes', type=int, default=100, help='episodes per map per density')
    parser.add_argument('--densities', type=str, default='4', help='comma-separated obstacle counts')
    parser.add_argument('--seed', type=int, default=0, help='base seed; episode i uses seed+i')
    parser.add_argument('--out', type=str, default='eval_results.csv')
    cfg = parser.parse_args()
    if cfg.max_steps <= 0 or cfg.episodes <= 0:
        parser.error('max_steps and episodes must be positive')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    device = torch.device('cuda:0' if torch.cuda.is_available() and cfg.gpu else 'cpu')
    densities = [int(d) for d in cfg.densities.split(',')]
    if any(d < 0 for d in densities):
        parser.error('densities must be non-negative')

    try:
        runner = TripRunner(cfg, device)
    except SettingsError as error:
        parser.error(str(error))

    rows = []
    with open(cfg.out, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for map_path in cfg.static_map:
            for n_dyn in densities:
                for i in range(cfg.episodes):
                    row = runner.run(map_path, n_dyn, cfg.seed + i)
                    writer.writerow(row)
                    fh.flush()
                    rows.append(row)
                    logging.info('%s | requested=%d actual=%s | seed=%d | %s | steps=%s',
                                 row['map'], n_dyn, row['n_dynamic'], row['seed'], row['outcome'],
                                 row['steps'])

    summarise(rows, cfg.out)


def summarise(rows, out_path):
    """Separate generation failures from executed episodes and their collision rates."""
    if not rows:
        logging.warning('No scenarios attempted.')
        return
    groups = {}
    failed = sum(r['outcome'] == 'scenario_failed' for r in rows)
    short = sum(r['outcome'] != 'scenario_failed' and
                r['n_dynamic'] < r['requested_n_dynamic'] for r in rows)
    for row in rows:
        if row['outcome'] != 'scenario_failed':
            groups.setdefault((row['map'], row['n_dynamic']), []).append(row)
    print('Metrics v2 | attempted={} executed={} scenario_failed={} underfilled={}'.format(
        len(rows), len(rows) - failed, failed, short))
    print('map n runs success coll_dyn coll_static coll_both coll_total timeout detour')
    for (map_name, n_dyn), group in sorted(groups.items()):
        total = len(group)
        def rate(name):
            return sum(r['outcome'] == name for r in group) / total
        coll = sum(r['outcome'].startswith('collision_') for r in group) / total
        detours = [r['detour_ratio'] for r in group
                   if r['outcome'] == 'success' and r['detour_ratio'] != '']
        print('{} {} {} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {}'.format(
            map_name, n_dyn, total, rate('success'), rate('collision_dynamic'),
            rate('collision_static'), rate('collision_both'), coll, rate('timeout'),
            '{:.3f}'.format(sum(detours) / len(detours)) if detours else '-'))
    print('Per-episode rows written to {}'.format(out_path))


if __name__ == '__main__':
    main()
