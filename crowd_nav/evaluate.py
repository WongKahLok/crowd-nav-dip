"""Batch evaluation of Theta* guidance and local avoidance (metrics version 2).

Run from crowd_nav/ with the project on PYTHONPATH:
    python evaluate.py --policy sarl --model_dir data/output-trained \
        --static_map ../static_maps/land_shapes_sf_crop.png --episodes 100 --densities 2,4,8

or on the frozen scenarios (static_maps/scenarios, one fixed episode per map each):
    python evaluate.py --policy sarl --model_dir data/output-trained --scenario P5 P8 P12 P15

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
import os

import gym
import numpy as np
import torch

from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.utils.land_map import load_static_map
from crowd_nav.utils import scenario as scenario_files
from crowd_nav.utils.local_goal import LocalGoalTracker, nearest_point_goal
from crowd_nav.utils.theta_star import ThetaStar, interpolate_path
from crowd_sim.envs.utils.info import Collision, ReachGoal, Timeout
from crowd_sim.envs.utils.robot import Robot

GRID_X, GRID_Y, GRID_RES = 50, 50, 1  # same discretisation test.py uses

CSV_FIELDS = ['map', 'seed', 'n_dynamic', 'outcome', 'steps', 'sim_time', 'travelled_dist',
              'planned_dist', 'detour_ratio', 'min_sep_dynamic', 'min_sep_static',
              'danger_frac', 'plan_cost', 'start_x', 'start_y', 'goal_x', 'goal_y',
              'requested_n_dynamic', 'collision_kind', 'metrics_version', 'emergency_frac',
              'scenario']


class EpisodeCancelled(Exception):
    """Raised out of an ``on_step`` callback to abandon the running episode."""


def build_grids(map_path, safety_distance):
    """Two grids from one map: raw for simulated geometry, inflated for planning (as test.py).

    ``map_path`` is a PNG land mask (read at GRID_X x GRID_Y) or a land_map.py JSON file,
    which carries its own size and resolution.
    """
    sim_grid = load_static_map(map_path, x_range=GRID_X, y_range=GRID_Y, resolution=GRID_RES,
                               safety_distance=0)
    plan_grid = load_static_map(map_path, x_range=GRID_X, y_range=GRID_Y, resolution=GRID_RES,
                                safety_distance=safety_distance)
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

    The same eight lines were open coded in ``main`` and in test.py; keeping them here
    means the GUI session and the batch evaluator cannot drift apart.
    ``args`` only has to carry ``colregs``, which ``CrowdSim_test.configure`` reads.
    """
    env = gym.make('CrowdSimTest-v0')
    env.configure(env_config, args)
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)
    env.set_obstacles(sim_grid.get_static_obstacles())
    env.set_init_para(sim_grid.get_env_para())
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
    elif getattr(cfg, 'legacy_local_goal', False):
        # the original stateless nearest-point goal, the "before" of the tracker
        tracker = _NearestPointGoal(path_dense, cfg.local_goal_dist, cfg.monitor_dist)
        robot.gx, robot.gy = tracker.update(robot.get_full_state(), ob)
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


class _NearestPointGoal(object):
    """``nearest_point_goal`` behind the tracker's ``update(state, ob)`` interface."""

    def __init__(self, path, goal_dist, monitor_dist):
        self.path, self.goal_dist, self.monitor_dist = path, goal_dist, monitor_dist

    def update(self, ship_state, ob):
        return nearest_point_goal(ship_state, ob, self.path, self.goal_dist, self.monitor_dist)


def scenario_failed_row(map_name, seed, requested):
    row = dict.fromkeys(CSV_FIELDS, '')
    row.update(map=map_name, seed=seed, requested_n_dynamic=requested,
               outcome='scenario_failed', metrics_version=2)
    return row


def build_policy(args, device):
    """Load the policy and its trained weights exactly the way test.py does."""
    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    if args.model_dir is not None:
        policy_config.read(os.path.join(args.model_dir, os.path.basename(args.policy_config)))
    else:
        policy_config.read(args.policy_config)
    policy.configure(policy_config)
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


def main():
    parser = argparse.ArgumentParser('Batch evaluation over randomised scenarios')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy', type=str, default='sarl')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--colregs', default=False, action='store_true')  # env.configure reads this
    parser.add_argument('--static_map', type=str, nargs='+', default=None,
                        help='maps to evaluate; with --scenario, defaults to every map it covers')
    parser.add_argument('--scenario', type=str, nargs='+', default=None,
                        help='frozen scenarios (P5 P8 P12 P15, or JSON paths) instead of random '
                             'sampling; each runs once per map, and --episodes, --densities, '
                             '--seed and --min_route_len are ignored')
    parser.add_argument('--episodes', type=int, default=100, help='episodes per map per density')
    parser.add_argument('--densities', type=str, default='4', help='comma-separated obstacle counts')
    parser.add_argument('--seed', type=int, default=0, help='base seed; episode i uses seed+i')
    parser.add_argument('--out', type=str, default='eval_results.csv')
    parser.add_argument('--no_global_plan', default=False, action='store_true',
                        help='B1 baseline: ignore the Theta* route and steer at the final goal')
    parser.add_argument('--legacy_local_goal', default=False, action='store_true',
                        help='"before" baseline: the original per-frame nearest-point local goal '
                             'instead of LocalGoalTracker')
    parser.add_argument('--local_goal_dist', type=float, default=1.0)
    parser.add_argument('--monitor_dist', type=float, default=5.0)
    parser.add_argument('--monitor_exit_dist', type=float, default=None,
                        help='obstacles stop being monitored beyond this, default 1.3 * monitor_dist')
    parser.add_argument('--lookahead_rate', type=float, default=1.0, help='m/s the look-ahead may change')
    parser.add_argument('--shore_clearance', type=float, default=0.5,
                        help='room between the ship-to-goal segment and the land, on top of the radius')
    parser.add_argument('--max_steps', type=int, default=500)  # 500 * 0.25s = 125s per episode
    parser.add_argument('--safety_distance', type=int, default=1)
    parser.add_argument('--min_route_len', type=float, default=20.0)
    cfg = parser.parse_args()
    if cfg.max_steps <= 0 or cfg.episodes <= 0:
        parser.error('max_steps and episodes must be positive')
    scenarios = []
    if cfg.scenario:
        try:
            scenarios = [scenario_files.load_scenario(name) for name in cfg.scenario]
            if cfg.static_map is None:
                cfg.static_map = list(dict.fromkeys(
                    scenario_files.map_path_of(entry)
                    for data in scenarios for entry in data['maps'].values()))
            for data in scenarios:           # fail before any episode runs
                for map_path in cfg.static_map:
                    scenario_files.entry_for(data, map_path)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    elif cfg.static_map is None:
        parser.error('--static_map is required without --scenario')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    device = torch.device('cuda:0' if torch.cuda.is_available() and cfg.gpu else 'cpu')
    densities = [int(d) for d in cfg.densities.split(',')]
    if any(d < 0 for d in densities):
        parser.error('densities must be non-negative')

    env_config = configparser.RawConfigParser()
    env_config.read(os.path.join(cfg.model_dir, os.path.basename(cfg.env_config))
                    if cfg.model_dir else cfg.env_config)
    policy = build_policy(cfg, device)

    rows = []
    with open(cfg.out, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for map_path in cfg.static_map:
            map_name = os.path.splitext(os.path.basename(map_path))[0]
            sim_grid, plan_grid = build_grids(map_path, cfg.safety_distance)
            cells = free_cells(plan_grid)

            # Each map gets its own environment and episode geometry.
            env, robot = build_env(env_config, cfg, policy, sim_grid)

            for data in scenarios:
                entry = scenario_files.entry_for(data, map_path)
                n_dyn = data['n_obstacles']
                row = run_episode(env, robot, sim_grid, scenario_files.to_episode(entry), cfg)
                row.update(map=map_name, seed=entry['seed'], requested_n_dynamic=n_dyn,
                           scenario=data['name'])
                if row['n_dynamic'] != n_dyn:
                    logging.warning('%s %s: the scenario has %d of %d obstacles',
                                    map_name, data['name'], row['n_dynamic'], n_dyn)
                writer.writerow(row)
                fh.flush()
                rows.append(row)
                logging.info('%s | scenario=%s actual=%s | %s | steps=%s', map_name,
                             data['name'], row['n_dynamic'], row['outcome'], row['steps'])

            for n_dyn in ([] if scenarios else densities):
                for i in range(cfg.episodes):
                    seed = cfg.seed + i
                    np.random.seed(seed)  # scenario sampling is the only stochastic part
                    scenario = sample_scenario(plan_grid, cells, n_dyn, cfg.min_route_len)
                    if scenario is None:
                        logging.warning('%s seed %d: no valid scenario found', map_name, seed)
                        row = scenario_failed_row(map_name, seed, n_dyn)
                    else:
                        row = run_episode(env, robot, sim_grid, scenario, cfg)
                        row.update(map=map_name, seed=seed, requested_n_dynamic=n_dyn)
                        if row['n_dynamic'] != n_dyn:
                            logging.warning('%s seed %d: placed %d of %d requested obstacles',
                                            map_name, seed, row['n_dynamic'], n_dyn)
                    writer.writerow(row)
                    fh.flush()
                    rows.append(row)
                    logging.info('%s | requested=%d actual=%s | seed=%d | %s | steps=%s',
                                 map_name, n_dyn, row['n_dynamic'], seed, row['outcome'], row['steps'])

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
