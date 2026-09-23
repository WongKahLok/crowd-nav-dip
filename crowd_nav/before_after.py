"""Before/after videos of the frozen scenarios, through the GUI's own export.

"Before" is the original per-frame nearest-point local goal
(:func:`crowd_nav.utils.local_goal.nearest_point_goal`); "after" is the current
:class:`~crowd_nav.utils.local_goal.LocalGoalTracker`. Everything else (the SARL
policy and its action filter, the map, the route, the obstacle ships) is identical, so
each pair shows what the local-goal change alone does.

    python -m crowd_nav.before_after [--scenarios P5 P8 P12 P15]
        [--map static_maps/land_shapes_sf_crop.png] [--out_dir videos/before_after]

Writes ``<scenario>_<map>_<before|after>.mp4`` (``.gif`` without ffmpeg) and a
``summary.csv`` with each episode's metrics.
"""
import argparse
import csv
import logging
import os

from crowd_nav.session import ScenarioSession
from crowd_nav.utils import scenario as scenario_files

REPO_ROOT = scenario_files.REPO_ROOT
MODES = (('before', True, 'original nearest-point local goal'),
         ('after', False, 'LocalGoalTracker'))
SUMMARY_FIELDS = ['scenario', 'map', 'mode', 'outcome', 'steps', 'sim_time', 'n_dynamic',
                  'travelled_dist', 'detour_ratio', 'min_sep_dynamic', 'min_sep_static',
                  'danger_frac', 'emergency_frac', 'video']


def main(argv=None):
    parser = argparse.ArgumentParser('Before/after videos of the frozen scenarios')
    parser.add_argument('--scenarios', nargs='+', default=list(scenario_files.PHASES))
    parser.add_argument('--map', default=os.path.join(REPO_ROOT, 'static_maps',
                                                      'land_shapes_sf_crop.png'))
    parser.add_argument('--model_dir', default=os.path.join(REPO_ROOT, 'crowd_nav', 'data',
                                                            'output-trained'))
    parser.add_argument('--policy', default='sarl')
    parser.add_argument('--out_dir', default=os.path.join(REPO_ROOT, 'videos', 'before_after'))
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--fps', type=int, default=8)
    parser.add_argument('--gpu', default=False, action='store_true')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    session = ScenarioSession(args.model_dir, policy_name=args.policy,
                              device='cuda:0' if args.gpu else 'cpu')
    session.load_map(args.map)
    map_name = scenario_files.map_key(args.map)
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for name in args.scenarios:
        for mode, legacy, label in MODES:
            session.legacy_local_goal = legacy
            entry = session.load_scenario(name)      # also clears the previous result
            result = session.run_episode(max_steps=args.max_steps)
            metrics = result.metrics
            title = '{} on {} ({} ships) - {}: {}\n{}, {} steps'.format(
                name, map_name, metrics['n_dynamic'], mode, label,
                metrics['outcome'], metrics['steps'])
            path = os.path.join(args.out_dir, '{}_{}_{}.mp4'.format(name, map_name, mode))
            written = session.export_video(result, path, fps=args.fps, title=title)
            logging.info('%s %s: %s in %d steps, seed %d -> %s', name, mode,
                         metrics['outcome'], metrics['steps'], entry['seed'], written)
            row = {k: metrics.get(k, '') for k in SUMMARY_FIELDS}
            row.update(scenario=name, map=map_name, mode=mode,
                       video=os.path.basename(written))
            rows.append(row)

    summary = os.path.join(args.out_dir, 'summary.csv')
    with open(summary, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print('scenario mode   outcome            steps  min_sep_dyn  min_sep_static  emergency')
    for r in rows:
        print('{:<8} {:<6} {:<18} {:>5}  {:>11}  {:>14}  {:>9}'.format(
            r['scenario'], r['mode'], r['outcome'], r['steps'], r['min_sep_dynamic'],
            r['min_sep_static'], r['emergency_frac']))
    print('Videos and summary.csv in {}'.format(args.out_dir))


if __name__ == '__main__':
    main()
