"""Batch evaluation harness for CrowdNav-DIP (TODO.md Phase 2).

Runs the same Theta* + rolling-local-goal + learned-policy loop that test.py runs, but over
many randomised, seeded scenarios, writing one CSV row per episode instead of a video.

test.py answers "does it work on this one hard-coded scenario?"; this answers "how often does
it work, and how well?" -- which is what the report needs.

Usage:
    python evaluate.py --policy sarl --model_dir data/output-trained \
                       --static_map ../static_maps/land_shapes_sf_crop.png \
                       --episodes 100 --densities 2,4,8 --out results.csv

The four baselines of TODO.md Phase 3 are flag combinations, all on identical seeds:
    B1  ORCA alone, no global planner   --policy orca   --no_global_plan
    B2  Theta* + pure pursuit           --policy linear
    B3  Theta* + ORCA                   --policy orca
    B4  Theta* + SARL (this system)     --policy sarl

Caveats worth knowing when interpreting the numbers:
  * Dynamic obstacles are ORCA-driven and only observe *each other* (crowd_sim.py:787), never
    the static map, so they sail straight through land. They are traffic, not a land model.
  * env.light_reset() zeroes global_time every tick, so the environment's own time_limit
    Timeout can never fire; --max_steps is the only timeout, and it is ours.
  * Collision attribution (static vs dynamic) compares post-step clearances, while step()
    detects collisions with a swept pre-step check, so attribution is a close approximation.
    
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
from crowd_nav.utils.theta_star import HybridEnv, ThetaStar, get_local_goal, interpolate_path
from crowd_sim.envs.utils.info import Collision, ReachGoal
from crowd_sim.envs.utils.robot import Robot

GRID_X, GRID_Y, GRID_RES = 50, 50, 1  # same discretisation test.py uses

CSV_FIELDS = ['map', 'seed', 'n_dynamic', 'outcome', 'steps', 'sim_time', 'travelled_dist',
              'planned_dist', 'detour_ratio', 'min_sep_dynamic', 'min_sep_static',
              'danger_frac', 'plan_cost', 'start_x', 'start_y', 'goal_x', 'goal_y']


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


def min_clearance(robot, agents):
    """Smallest surface-to-surface gap between the robot and any agent (inf if none)."""
    gap = float('inf')
    for agent in agents:
        d = math.dist((robot.px, robot.py), (agent.px, agent.py)) - robot.radius - agent.radius
        gap = min(gap, d)
    return gap


def run_episode(env, robot, sim_grid, scenario, cfg):
    """One episode of the full stack. Returns a metrics dict (one CSV row)."""
    env.set_local_goal_list(sim_grid.grid_2_drl(scenario['path']))
    ob = env.reset('test', None, scenario['dyn'])
    path_dense = interpolate_path(env.global_path)[1:]
    planned = sum(math.dist(path_dense[i], path_dense[i + 1]) for i in range(len(path_dense) - 1))

    if cfg.no_global_plan:
        # B1: no global guidance -- aim straight at the final goal for the whole episode.
        # The Theta* route is still planned, but only as the optimal-distance reference.
        robot.gx, robot.gy = path_dense[-1][0], path_dense[-1][1]
    else:
        one_goal = get_local_goal(robot.get_full_state(), ob, path_dense,
                                  cfg.local_goal_dist, cfg.monitor_dist)
        robot.gx, robot.gy = one_goal[0], one_goal[1]

    travelled, danger_steps, steps = 0.0, 0, 0
    min_sep_dyn, min_sep_stat = float('inf'), float('inf')
    outcome = 'timeout'  # stays timeout unless something below overrides it

    for steps in range(1, cfg.max_steps + 1):
        ob = env.light_reset('test', None)  # dynamic humans + static cells within nearby_dist
        action = robot.act(ob)  # policy scores every candidate action, picks the best
        prev = (robot.px, robot.py)
        ob, _, done, info = env.step(action)
        travelled += math.dist(prev, (robot.px, robot.py))

        # get_humans_nearby() appends dynamic humans first, then nearby static cells
        nearby = env.get_humans_nearby()
        d_dyn = min_clearance(robot, nearby[:len(env.humans)])
        d_stat = min_clearance(robot, nearby[len(env.humans):])
        min_sep_dyn, min_sep_stat = min(min_sep_dyn, d_dyn), min(min_sep_stat, d_stat)
        if min(d_dyn, d_stat) < env.discomfort_dist:
            danger_steps += 1

        if isinstance(info, Collision):
            outcome = 'collision_dynamic' if d_dyn <= d_stat else 'collision_static'
            break
        # ReachGoal can only fire at the true path end: get_local_goal never returns a point
        # closer than local_goal_dist (1 m) unless the path has run out, and robot.radius < 1 m
        if isinstance(info, ReachGoal) or \
                math.dist((robot.px, robot.py), path_dense[-1]) < robot.radius:
            outcome = 'success'
            break

        if not cfg.no_global_plan:  # B1 keeps aiming at the final goal, so nothing to slide
            one_goal = get_local_goal(robot.get_full_state(), ob, path_dense,
                                      cfg.local_goal_dist, cfg.monitor_dist)
            robot.gx, robot.gy = one_goal[0], one_goal[1]  # slide the local goal forward each tick

    return {
        'outcome': outcome,
        'steps': steps,
        'sim_time': round(steps * env.time_step, 2),
        'travelled_dist': round(travelled, 3),
        'planned_dist': round(planned, 3),
        'detour_ratio': round(travelled / planned, 3) if planned > 0 else '',
        'min_sep_dynamic': round(min_sep_dyn, 3) if math.isfinite(min_sep_dyn) else '',
        'min_sep_static': round(min_sep_stat, 3) if math.isfinite(min_sep_stat) else '',
        'danger_frac': round(danger_steps / steps, 3) if steps else '',
        'plan_cost': round(scenario['cost'], 3),
        'start_x': scenario['start'][0], 'start_y': scenario['start'][1],
        'goal_x': scenario['goal'][0], 'goal_y': scenario['goal'][1],
    }


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
    parser.add_argument('--static_map', type=str, nargs='+', required=True)
    parser.add_argument('--episodes', type=int, default=100, help='episodes per map per density')
    parser.add_argument('--densities', type=str, default='4', help='comma-separated obstacle counts')
    parser.add_argument('--seed', type=int, default=0, help='base seed; episode i uses seed+i')
    parser.add_argument('--out', type=str, default='eval_results.csv')
    parser.add_argument('--no_global_plan', default=False, action='store_true',
                        help='B1 baseline: ignore the Theta* route and steer at the final goal')
    parser.add_argument('--local_goal_dist', type=float, default=1.0)
    parser.add_argument('--monitor_dist', type=float, default=5.0)
    parser.add_argument('--max_steps', type=int, default=500)  # 500 * 0.25s = 125s per episode
    parser.add_argument('--safety_distance', type=int, default=1)
    parser.add_argument('--min_route_len', type=float, default=20.0)
    cfg = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    device = torch.device('cuda:0' if torch.cuda.is_available() and cfg.gpu else 'cpu')
    densities = [int(d) for d in cfg.densities.split(',')]

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

            # one env per map: set_obstacles() appends, so it must not be called twice per map
            env = gym.make('CrowdSimTest-v0')
            env.configure(env_config, cfg)
            robot = Robot(env_config, 'robot')
            robot.set_policy(policy)
            env.set_robot(robot)
            env.set_obstacles(sim_grid.get_static_obstacles())
            env.set_init_para(sim_grid.get_env_para())
            policy.set_env(env)

            for n_dyn in densities:
                for i in range(cfg.episodes):
                    seed = cfg.seed + i
                    np.random.seed(seed)  # scenario sampling is the only stochastic part
                    scenario = sample_scenario(plan_grid, cells, n_dyn, cfg.min_route_len)
                    if scenario is None:
                        logging.warning('%s seed %d: no valid scenario found, skipping', map_name, seed)
                        continue
                    row = run_episode(env, robot, sim_grid, scenario, cfg)
                    row.update({'map': map_name, 'seed': seed, 'n_dynamic': n_dyn})
                    writer.writerow(row)
                    fh.flush()  # keep partial results if a long run is interrupted
                    rows.append(row)
                    logging.info('%s | n=%d | seed=%3d | %-17s | %3d steps | detour %s',
                                 map_name, n_dyn, seed, row['outcome'], row['steps'],
                                 row['detour_ratio'])

    summarise(rows, cfg.out)


def summarise(rows, out_path):
    """Print success/collision/timeout rates per (map, density) -- the table for the report."""
    if not rows:
        logging.warning('No episodes completed.')
        return
    groups = {}
    for row in rows:
        groups.setdefault((row['map'], row['n_dynamic']), []).append(row)

    print('\n%-28s %5s %6s %8s %9s %8s %9s %8s' %
          ('map', 'n', 'runs', 'success', 'coll_dyn', 'coll_st', 'timeout', 'detour'))
    for (map_name, n_dyn), group in sorted(groups.items()):
        total = len(group)
        rate = lambda name: sum(r['outcome'] == name for r in group) / total
        detours = [r['detour_ratio'] for r in group
                   if r['outcome'] == 'success' and r['detour_ratio'] != '']
        print('%-28s %5d %6d %8.2f %9.2f %8.2f %9.2f %8s' %
              (map_name, n_dyn, total, rate('success'), rate('collision_dynamic'),
               rate('collision_static'), rate('timeout'),
               '%.3f' % (sum(detours) / len(detours)) if detours else '-'))
    print('\nPer-episode rows written to %s' % out_path)


if __name__ == '__main__':
    main()
