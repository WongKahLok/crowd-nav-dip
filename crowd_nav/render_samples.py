"""Render example episodes of an evaluated arm as mp4 videos, one per kind of outcome.

Reads <results>/<arm>_<map>.csv (written by run_arm.sh), picks representative episodes,
replays each one with the same settings and saves a video. Episodes replay exactly:
the seed fixes the scenario and every policy is deterministic in the test phase. Each
replay is checked against its CSV row (outcome and step count) before it is rendered,
so a video always shows the episode that was measured.

Normally called through render_arm.sh with the same settings as the run:

    POLICY=orca ./crowd_nav/render_arm.sh b3_theta_orca
    ./crowd_nav/render_arm.sh b4_theta_sarl --dry_run          # list the picks only
    ./crowd_nav/render_arm.sh b4_theta_sarl --types collision_dynamic --per_type 3
    ./crowd_nav/render_arm.sh b4_theta_sarl --episodes ht:16:2 ht:16:6   # chosen episodes

Kinds of episode (--types):
    success_typical    success whose duration is closest to the median success
    success_closest    success with the smallest gap to a moving ship (near miss)
    longest_detour     success with the largest travelled / planned distance
    most_fallback      episode where the emergency fallback acted most (SARL only)
    collision_dynamic  hit a moving ship
    collision_static   hit land or the map edge
    collision_both     hit both within one step
    timeout            ran out of time
"""
import argparse
import configparser
import csv
import logging
import os
import statistics

import matplotlib.pyplot as plt
import numpy as np
import torch

from crowd_nav.evaluate import (build_env, build_grids, build_policy, free_cells,
                                run_episode, sample_scenario)

MAPS = {'sf': 'static_maps/land_shapes_sf_crop.png', 'ht': 'static_maps/land_shapes_ht_crop.png'}
TYPES = ['success_typical', 'success_closest', 'longest_detour', 'most_fallback',
         'collision_dynamic', 'collision_static', 'collision_both', 'timeout']


def parse_args():
    ap = argparse.ArgumentParser(description='Render example episodes of an evaluated arm')
    ap.add_argument('--arm', required=True, help='arm name given to run_arm.sh')
    ap.add_argument('--results', default='crowd_nav/results-v2', help='folder holding the arm CSVs')
    ap.add_argument('--videos', default=None, help='output folder (default <results>/videos/<arm>)')
    ap.add_argument('--types', nargs='+', default=TYPES, choices=TYPES)
    ap.add_argument('--per_type', type=int, default=1, help='episodes per kind per map')
    ap.add_argument('--maps', nargs='+', default=['sf', 'ht'], choices=sorted(MAPS))
    ap.add_argument('--ships', type=int, nargs='+', default=None, help='only these requested densities')
    ap.add_argument('--episodes', nargs='+', default=None, metavar='MAP:SHIPS:SEED',
                    help='render exactly these episodes instead of one per kind, e.g. ht:16:2 sf:8:36')
    ap.add_argument('--dry_run', action='store_true', help='list the chosen episodes without rendering')
    # the same settings evaluate.py takes; they must match the run
    ap.add_argument('--env_config', type=str, default='configs/env.config')
    ap.add_argument('--policy_config', type=str, default='configs/policy.config')
    ap.add_argument('--policy', type=str, default='sarl')
    ap.add_argument('--model_dir', type=str, default='crowd_nav/data/output-trained')
    ap.add_argument('--il', default=False, action='store_true')
    ap.add_argument('--gpu', default=False, action='store_true')
    ap.add_argument('--colregs', default=False, action='store_true')
    ap.add_argument('--no_global_plan', default=False, action='store_true')
    ap.add_argument('--local_goal_dist', type=float, default=1.0)
    ap.add_argument('--monitor_dist', type=float, default=5.0)
    ap.add_argument('--monitor_exit_dist', type=float, default=None)
    ap.add_argument('--lookahead_rate', type=float, default=1.0)
    ap.add_argument('--shore_clearance', type=float, default=0.5)
    ap.add_argument('--max_steps', type=int, default=500)
    ap.add_argument('--safety_distance', type=int, default=1)
    ap.add_argument('--nearby_dist', type=float, default=None)
    ap.add_argument('--min_route_len', type=float, default=20.0)
    return ap.parse_args()


def num(row, field):
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError):
        return None


def pick(rows, kind, n, taken):
    """Up to n episodes of one kind, skipping episodes another kind already took."""
    done = [r for r in rows if r['outcome'] == 'success']
    if kind == 'success_typical':
        if not done:
            return []
        middle = statistics.median(num(r, 'steps') for r in done)
        ranked = sorted(done, key=lambda r: (abs(num(r, 'steps') - middle), int(r['seed'])))
    elif kind == 'success_closest':
        ranked = sorted((r for r in done if num(r, 'min_sep_dynamic') is not None),
                        key=lambda r: num(r, 'min_sep_dynamic'))
    elif kind == 'longest_detour':
        ranked = sorted((r for r in done if num(r, 'detour_ratio') is not None),
                        key=lambda r: -num(r, 'detour_ratio'))
    elif kind == 'most_fallback':
        ranked = sorted((r for r in rows if (num(r, 'emergency_frac') or 0) > 0),
                        key=lambda r: -num(r, 'emergency_frac'))
    else:
        ranked = sorted((r for r in rows if r['outcome'] == kind),
                        key=lambda r: (int(r['requested_n_dynamic']), int(r['seed'])))
    chosen = []
    for r in ranked:
        key = (r['requested_n_dynamic'], r['seed'])
        if key in taken:
            continue
        taken.add(key)
        chosen.append(r)
        if len(chosen) == n:
            break
    return chosen


def render(env, path, title):
    """env.render('video') with a caption: the figure it creates gets a title."""
    original = plt.subplots
    if len(title) > 72:                       # keep a long caption inside the frame
        parts = title.split(' | ')
        half = max(len(parts) // 2, 1)
        title = ' | '.join(parts[:half]) + '\n' + ' | '.join(parts[half:])

    def titled(*args, **kwargs):
        fig, ax = original(*args, **kwargs)
        fig.suptitle(title, fontsize=10)
        return fig, ax

    plt.subplots = titled
    try:
        env.render('video', path)
    finally:
        plt.subplots = original
        plt.close('all')


def main():
    args = parse_args()
    logging.basicConfig(level=logging.WARNING)
    torch.set_num_threads(1)  # match run_arm.sh so replays are bit-identical
    device = torch.device('cpu')
    out_dir = args.videos or os.path.join(args.results, 'videos', args.arm)

    env_config = configparser.RawConfigParser()
    env_config.read(os.path.join(args.model_dir, os.path.basename(args.env_config)))
    policy = build_policy(args, device)

    rendered, mismatched = [], []
    for m in args.maps:
        csv_path = os.path.join(args.results, '{}_{}.csv'.format(args.arm, m))
        if not os.path.exists(csv_path):
            print('{}: {} not found, skipped'.format(m.upper(), csv_path))
            continue
        with open(csv_path, newline='') as fh:
            rows = [r for r in csv.DictReader(fh) if r['outcome'] != 'scenario_failed']
        if args.ships:
            rows = [r for r in rows if int(r['requested_n_dynamic']) in args.ships]
        taken, picks = set(), []
        print('{}: {} episodes in {}'.format(m.upper(), len(rows), csv_path))
        if args.episodes:
            wanted = [e.split(':') for e in args.episodes if e.split(':')[0] == m]
            for _, ships, seed in wanted:
                match = [r for r in rows if r['requested_n_dynamic'] == ships and int(r['seed']) == int(seed)]
                if not match:
                    print('  {} ships, seed {}: not in this arm'.format(ships, seed))
                picks += [('episode', r) for r in match]
        else:
            for kind in args.types:
                chosen = pick(rows, kind, args.per_type, taken)
                if not chosen:
                    print('  {:18s} none in this arm'.format(kind))
                picks += [(kind, r) for r in chosen]
        for kind, r in picks:
            print('  {:18s} {:>2s} ships, seed {:3d}: {} after {} steps'.format(
                kind, r['requested_n_dynamic'], int(r['seed']), r['outcome'], r['steps']))
        if args.dry_run or not picks:
            continue

        sim_grid, plan_grid = build_grids(MAPS[m], args.safety_distance)
        cells = free_cells(plan_grid)
        env, robot = build_env(env_config, args, policy, sim_grid)
        os.makedirs(out_dir, exist_ok=True)

        for kind, r in picks:
            seed, ships = int(r['seed']), int(r['requested_n_dynamic'])
            np.random.seed(seed)  # the same scenario evaluate.py sampled
            scenario = sample_scenario(plan_grid, cells, ships, args.min_route_len)
            result = run_episode(env, robot, sim_grid, scenario, args)
            if result['outcome'] != r['outcome'] or str(result['steps']) != r['steps']:
                mismatched.append((m, kind, seed))
                print('  ! {} seed {}: the replay gave {} after {} steps but the CSV says {} after {}. '
                      'Not rendered: use the same POLICY, MODEL_DIR and flags as the run.'.format(
                          kind, seed, result['outcome'], result['steps'], r['outcome'], r['steps']))
                continue
            path = os.path.join(out_dir, '{}_{}_{}ships_seed{}.mp4'.format(m, kind, ships, seed))
            title = '{} | {} | {} ships | seed {} | {} after {:.2f} s'.format(
                args.arm, m.upper(), ships, seed, r['outcome'], result['sim_time'])
            render(env, path, title)
            rendered.append(path)
            print('  wrote {}'.format(path))

    if not args.dry_run:
        print('\n{} video(s) in {}'.format(len(rendered), out_dir))
        if mismatched:
            print('{} episode(s) did not replay identically and were skipped: {}'.format(
                len(mismatched), mismatched))


if __name__ == '__main__':
    main()
