"""Measure COLREGs compliance of an arm that has already been evaluated.

Replays episodes from <results>/<arm>_<map>.csv and feeds every executed action to
``colregs.EncounterTracker``, which classifies the encounters with Rule 13(d) memory and
scores the action taken in each one. Nothing about the policy changes: this is the
"before" measurement that a COLREGs-aware policy has to beat.

    ./crowd_nav/measure_compliance.sh b4_theta_sarl                  # 25 episodes per map
    ./crowd_nav/measure_compliance.sh b4_theta_sarl --sample 0       # every episode (slow)
    ./crowd_nav/measure_compliance.sh b4_theta_sarl --videos 3       # plus the worst 3 as mp4

Writes one row per episode to <results>/compliance/<arm>_<map>.csv and prints the totals.
Videos go to <results>/videos/<arm>-compliance/ with the violation times in the caption.
"""
import argparse
import configparser
import csv
import logging
import os

import numpy as np
import torch

from crowd_nav.evaluate import (build_env, build_grids, build_policy, free_cells,
                                run_episode, sample_scenario)
from crowd_nav.policy import colregs
from crowd_nav.render_samples import MAPS, render
from crowd_sim.envs.utils.action import ActionXY

KINDS = [colregs.HEAD_ON, colregs.CROSSING, colregs.OVERTAKING, colregs.OVERTAKEN]
FIELDS = (['map', 'seed', 'ships', 'outcome', 'steps', 'encounters', 'compliant_encounters',
           'compliance_rate', 'violating_steps', 'encounter_steps', 'stopped_steps']
          + ['{}_encounters'.format(k) for k in KINDS] + ['{}_compliant'.format(k) for k in KINDS])


def parse_args():
    ap = argparse.ArgumentParser(description='COLREGs compliance of an evaluated arm')
    ap.add_argument('--arm', required=True)
    ap.add_argument('--results', default='crowd_nav/results-v2')
    ap.add_argument('--maps', nargs='+', default=['sf', 'ht'], choices=sorted(MAPS))
    ap.add_argument('--sample', type=int, default=25, help='episodes per map, 0 for all')
    ap.add_argument('--ships', type=int, nargs='+', default=None, help='only these densities')
    ap.add_argument('--videos', type=int, default=0, help='render this many of the worst episodes')
    # rule thresholds, all of them interpretations that belong in the report
    ap.add_argument('--safe_cpa', type=float, default=1.0, help='closest approach that counts as clear, m')
    ap.add_argument('--max_range', type=float, default=15.0, help='ignore vessels beyond this, m')
    ap.add_argument('--max_tcpa', type=float, default=30.0, help='ignore approaches further off than this, s')
    ap.add_argument('--clear_range', type=float, default=colregs.CLEAR_RANGE,
                    help='an encounter ends beyond this range while opening, m')
    # the settings of the run being replayed; they must match it
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


def spread(rows, sample):
    """Take `sample` rows spread evenly over the arm, so every density is represented."""
    rows = sorted(rows, key=lambda r: (int(r['requested_n_dynamic']), int(r['seed'])))
    if not sample or sample >= len(rows):
        return rows
    step = len(rows) / float(sample)
    return [rows[int(i * step)] for i in range(sample)]


def watch(env, robot, tracker, violations):
    """An on_step callback that scores the action just executed against the rules."""
    def on_step(steps, _max_steps):
        own, ships = env.states[-1][0], env.states[-1][1]   # the state the action was chosen in
        action = ActionXY(robot.vx, robot.vy)               # holonomic: velocity is the action
        for key, encounter in tracker.update(own, ships, action=action).items():
            if colregs.violates(action, encounter, own, ships[key], safe_cpa=tracker.safe_cpa):
                violations.append((steps, encounter.kind, encounter.role))
    return on_step


def measure(env, robot, sim_grid, plan_grid, cells, row, args):
    """Replay one episode and return its compliance row plus the violating steps."""
    seed, ships = int(row['seed']), int(row['requested_n_dynamic'])
    np.random.seed(seed)
    scenario = sample_scenario(plan_grid, cells, ships, args.min_route_len)
    tracker = colregs.EncounterTracker(clear_range=args.clear_range, safe_cpa=args.safe_cpa,
                                       max_range=args.max_range, max_tcpa=args.max_tcpa)
    violations = []
    result = run_episode(env, robot, sim_grid, scenario, args,
                         on_step=watch(env, robot, tracker, violations))
    summary = tracker.summary()
    record = {
        'map': row['map'], 'seed': seed, 'ships': ships, 'outcome': result['outcome'],
        'steps': result['steps'], 'encounters': summary['encounters'],
        'compliant_encounters': summary['compliant'],
        'compliance_rate': ('' if summary['compliance_rate'] is None
                            else round(summary['compliance_rate'], 4)),
        'violating_steps': summary['violating_steps'], 'encounter_steps': summary['steps'],
        'stopped_steps': summary['stopped_steps'],
    }
    for kind in KINDS:
        entry = summary['by_kind'].get(kind, {'encounters': 0, 'compliant': 0})
        record['{}_encounters'.format(kind)] = entry['encounters']
        record['{}_compliant'.format(kind)] = entry['compliant']
    matched = result['outcome'] == row['outcome'] and str(result['steps']) == row['steps']
    return record, violations, matched


def times(violations, time_step=0.25):
    """Violating steps as compact second ranges, for a video caption."""
    if not violations:
        return ''
    steps = sorted({s for s, _, _ in violations})
    spans, start, previous = [], steps[0], steps[0]
    for s in steps[1:]:
        if s != previous + 1:
            spans.append((start, previous))
            start = s
        previous = s
    spans.append((start, previous))
    return ', '.join('{:.1f}-{:.1f}s'.format(a * time_step, b * time_step) for a, b in spans[:4])


def main():
    args = parse_args()
    logging.basicConfig(level=logging.WARNING)
    torch.set_num_threads(1)
    env_config = configparser.RawConfigParser()
    env_config.read(os.path.join(args.model_dir, os.path.basename(args.env_config)))
    policy = build_policy(args, torch.device('cpu'))
    out_dir = os.path.join(args.results, 'compliance')
    os.makedirs(out_dir, exist_ok=True)

    everything, mismatched = [], 0
    for m in args.maps:
        csv_path = os.path.join(args.results, '{}_{}.csv'.format(args.arm, m))
        if not os.path.exists(csv_path):
            print('{}: {} not found, skipped'.format(m.upper(), csv_path))
            continue
        with open(csv_path, newline='') as fh:
            rows = [r for r in csv.DictReader(fh) if r['outcome'] != 'scenario_failed']
        if args.ships:
            rows = [r for r in rows if int(r['requested_n_dynamic']) in args.ships]
        rows = spread(rows, args.sample)
        print('{}: replaying {} episodes'.format(m.upper(), len(rows)))

        sim_grid, plan_grid = build_grids(MAPS[m], args.safety_distance)
        cells = free_cells(plan_grid)
        env, robot = build_env(env_config, args, policy, sim_grid)
        measured = []
        for i, row in enumerate(rows, 1):
            record, violations, matched = measure(env, robot, sim_grid, plan_grid, cells, row, args)
            mismatched += not matched
            measured.append((record, violations))
            print('  [{:3d}/{}] seed {:3d}, {} ships: {:2d} encounters, {:2d} violated, {}'.format(
                i, len(rows), record['seed'], record['ships'], record['encounters'],
                record['encounters'] - record['compliant_encounters'],
                'clean' if not violations else times(violations)))

        with open(os.path.join(out_dir, '{}_{}.csv'.format(args.arm, m)), 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            for record, _ in measured:
                writer.writerow(record)
        everything += [(m, record, violations) for record, violations in measured]

        if args.videos:
            worst = sorted(measured, key=lambda rv: (-len(rv[1]), -rv[0]['encounters']))[:args.videos]
            video_dir = os.path.join(args.results, 'videos', args.arm + '-compliance')
            os.makedirs(video_dir, exist_ok=True)
            for record, violations in worst:
                if not record['encounters']:
                    continue
                np.random.seed(record['seed'])
                scenario = sample_scenario(plan_grid, cells, record['ships'], args.min_route_len)
                run_episode(env, robot, sim_grid, scenario, args)
                path = os.path.join(video_dir, '{}_{}ships_seed{}.mp4'.format(
                    m, record['ships'], record['seed']))
                caption = '{} | {} | {} ships | seed {} | {} of {} encounters violated'.format(
                    args.arm, m.upper(), record['ships'], record['seed'],
                    record['encounters'] - record['compliant_encounters'], record['encounters'])
                if violations:
                    caption += ' | breaches at ' + times(violations)
                render(env, path, caption)
                print('  wrote {}'.format(path))

    if not everything:
        return
    total = lambda field: sum(r[field] for _, r, _ in everything)
    encounters, compliant = total('encounters'), total('compliant_encounters')
    print('\n=== {} : COLREGs compliance over {} episodes ==='.format(args.arm, len(everything)))
    print('episodes with at least one encounter: {} of {}'.format(
        sum(1 for _, r, _ in everything if r['encounters']), len(everything)))
    print('encounters: {}  compliant: {}  compliance rate: {}'.format(
        encounters, compliant,
        '{:.1%}'.format(compliant / encounters) if encounters else 'n/a (no encounters)'))
    print('steps inside an encounter: {}  violating: {}  stopped: {}'.format(
        total('encounter_steps'), total('violating_steps'), total('stopped_steps')))
    print('\n{:12s} {:>11s} {:>10s} {:>7s}'.format('rule', 'encounters', 'compliant', 'rate'))
    for kind in KINDS:
        n = total('{}_encounters'.format(kind))
        ok = total('{}_compliant'.format(kind))
        print('{:12s} {:11d} {:10d} {:>7s}'.format(
            kind, n, ok, '{:.0%}'.format(ok / n) if n else '-'))
    print('\nper-episode rows in {}'.format(out_dir))
    if mismatched:
        print('WARNING: {} replay(s) did not match the CSV; are the settings the same as the '
              'run?'.format(mismatched))


if __name__ == '__main__':
    main()
