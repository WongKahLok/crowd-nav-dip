"""One command for the experiments and videos of the project.

    python -m crowd_nav.exp demo --compare          one trip filmed without and with the rules
    python -m crowd_nav.exp colregs                 the report, running what is missing
    python -m crowd_nav.exp run colregs             sail one experiment of experiments.txt
    python -m crowd_nav.exp video colregs ht:8:60   film a trip (map:ships:seed)

Use the crowdnav environment and run it from the repository root. Add -h to a command
for its options.
"""
import argparse
import csv
import datetime
import functools
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import namedtuple

os.environ.setdefault('MPLBACKEND', 'Agg')  # videos are drawn offscreen

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE = os.path.join(REPO_ROOT, 'crowd_nav', 'experiments.txt')
MODELS = os.path.join(REPO_ROOT, 'crowd_nav', 'data')
RESULTS = os.path.join(REPO_ROOT, 'crowd_nav', 'results-v2')
MAP_KEYS = ('sf', 'ht')
if not __package__:  # started as a file: make the project importable
    sys.path.insert(0, REPO_ROOT)

# Trips per map per ship count. Every experiment sails seeds 0..TRIPS-1, so any two of
# them face the same situations.
TRIPS = 100

# the trip `demo` films when it is given nothing
DEMO = {'experiment': 'colregs', 'against': 'sarl', 'map': 'ht', 'ships': 8, 'seed': 60}

LABELS = {'sarl': 'SARL', 'sarl_colregs': 'SARL + COLREGs'}

# The pairs of the report: the same trips without the rule filter, then with it.
COLREGS = [('sarl', 'colregs'), ('sarl_dense', 'colregs_dense')]

Experiment = namedtuple('Experiment', 'name policy model ships extra')


class Problem(Exception):
    """Something the user can fix. Shown as one message, without a stack trace."""


USER_ERRORS = [Problem]


# -- the experiments table ------------------------------------------------------------

def load_table(path=TABLE):
    """crowd_nav/experiments.txt as {name: Experiment}, in the order of the file."""
    experiments = {}
    with open(path) as fh:
        for number, line in enumerate(fh, 1):
            fields = line.split('#')[0].split()
            if not fields:
                continue
            where = '{} line {}'.format(os.path.relpath(path), number)
            if len(fields) < 5:
                raise Problem('{}: expected "name policy model sf-ships ht-ships [settings]"'
                              .format(where))
            name, policy, model, sf, ht = fields[:5]
            if name in experiments:
                raise Problem('{}: the experiment "{}" is listed twice'.format(where, name))
            try:
                ships = {'sf': ship_counts(sf), 'ht': ship_counts(ht)}
            except ValueError:
                raise Problem('{}: ship counts are comma separated numbers such as 4,8,12, or - '
                              'to leave a map out'.format(where))
            experiments[name] = Experiment(name, policy, model, ships, fields[5:])
    return experiments


def ship_counts(text):
    return [] if text == '-' else [int(n) for n in text.split(',')]


def experiment(name):
    table = load_table()
    if name not in table:
        raise Problem('No experiment called "{}". The table has: {}'.format(name, ', '.join(table)))
    return table[name]


def engine():
    """The simulation stack, imported on first use so that -h works without it."""
    try:
        from crowd_nav import evaluate
    except ImportError as error:
        raise Problem('Cannot start the simulator: {}.\n{}'.format(error, environment_hint()))
    if evaluate.SettingsError not in USER_ERRORS:
        USER_ERRORS.append(evaluate.SettingsError)
    return evaluate


def environment_hint():
    """What is wrong with this Python, for when the simulator cannot start."""
    import importlib
    missing = []
    for module in ('numpy', 'torch', 'gym', 'matplotlib', 'PIL', 'rvo2', 'crowd_sim'):
        try:
            importlib.import_module(module)
        except Exception:   # a broken install fails in many ways
            missing.append(module)
    lines = ['This is {} (Python {}).'.format(sys.executable, platform.python_version())]
    if missing:
        lines.append('Missing here: {}.'.format(', '.join(missing)))
        lines.append('Activate the project environment first: conda activate crowdnav')
        lines.append('Otherwise follow Setup in README.md.')
    return '\n'.join(lines)


def settings(exp):
    """The run settings of an experiment, straight from its row of the table."""
    model_dir = os.path.join(MODELS, exp.model)
    if not os.path.isdir(model_dir):
        raise Problem('The experiment "{}" uses the model folder {}, which does not exist.'
                      .format(exp.name, model_dir))
    return engine().run_settings(exp.policy, model_dir, exp.extra)


def csv_path(results, name, m):
    return os.path.join(results, '{}_{}.csv'.format(name, m))


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


def shown(path):
    """A path as short as it can be printed: relative when it is below the current folder."""
    relative = os.path.relpath(path)
    return path if relative.startswith('..') else relative


# -- run --------------------------------------------------------------------------------

class Progress(object):
    """Prints how a batch of trips is getting on: one live line in a terminal, a line
    every half minute when the output goes to a file."""

    def __init__(self):
        self.started = time.time()
        self.last = 0.0
        self.outcomes = {}
        self.live = sys.stdout.isatty()

    def __call__(self, done, total, row):
        kind = row['outcome'].split('_')[0] if row['outcome'] != 'scenario_failed' else 'no scenario'
        self.outcomes[kind] = self.outcomes.get(kind, 0) + 1
        now = time.time()
        if done < total and now - self.last < (1.0 if self.live else 30.0):
            return
        self.last = now
        gone = now - self.started
        left = gone / done * (total - done)
        text = '  {}/{} trips | {} | {} gone, about {} left'.format(
            done, total, ', '.join('{} {}'.format(n, k) for k, n in sorted(self.outcomes.items())),
            minutes(gone), minutes(left))
        if self.live:
            print('\r' + text.ljust(100), end='\n' if done == total else '', flush=True)
        else:
            print(text, flush=True)


def minutes(seconds):
    return '{:d} s'.format(int(seconds)) if seconds < 90 else '{:.0f} min'.format(seconds / 60)


def outcome_table(rows):
    """Counts of each outcome per map and ship count."""
    map_key = engine().map_key
    cells = {}
    for r in rows:
        cells.setdefault((map_key(r['map']), int(r['requested_n_dynamic'])), []).append(r)
    print('{:5s} {:>5s} {:>4s} {:>8s} {:>9s} {:>8s} {:>12s} {:>12s}'.format(
        'map', 'ships', 'n', 'success', 'collision', 'timeout', 'no scenario', 'fewer ships'))
    for (m, ships), group in sorted(cells.items()):
        count = lambda test: sum(1 for r in group if test(r['outcome']))
        # the map had no room for every ship that was asked for; SF fills up at about 12
        short = sum(1 for r in group if r['n_dynamic'] != '' and int(r['n_dynamic']) < ships)
        print('{:5s} {:5d} {:4d} {:8d} {:9d} {:8d} {:12d} {:12d}'.format(
            m, ships, len(group), count(lambda o: o == 'success'),
            count(lambda o: o.startswith('collision')), count(lambda o: o == 'timeout'),
            count(lambda o: o == 'scenario_failed'), short))


def git(*command):
    try:
        return subprocess.run(('git', '-C', REPO_ROOT) + command, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ''


def run_record(exp, cfg, args, rules, started, seconds):
    """What a result has to be traceable to: settings, code and weights."""
    import torch
    weights = os.path.join(cfg.model_dir, 'rl_model.pth')
    digest = ''
    if os.path.exists(weights):
        with open(weights, 'rb') as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()[:16]
    return {
        'experiment': exp.name, 'policy': exp.policy, 'model': exp.model,
        'settings': list(exp.extra), 'ships': exp.ships, 'trips': args.trips,
        'seeds': '0-{}'.format(args.trips - 1), 'max_steps': cfg.max_steps,
        'rule_check': dict(rules, trips='every trip'),
        'started': started.strftime('%Y-%m-%d %H:%M:%S'), 'minutes': round(seconds / 60, 1),
        'jobs': args.jobs, 'commit': git('rev-parse', '--short', 'HEAD'),
        # tracked files that differ from that commit, or code that is not in git at all
        'local_changes': bool(git('status', '--porcelain', '--untracked-files=no')
                              or git('ls-files', '--others', '--exclude-standard', '--', '*.py')),
        'weights_sha256': digest, 'python': platform.python_version(), 'torch': torch.__version__,
    }


def has_results(results, exp):
    return any(os.path.exists(csv_path(results, exp.name, m)) for m in MAP_KEYS if exp.ships[m])


def run_experiment(exp, args):
    """Sail every trip of an experiment and write its results.

    The rule watch looks on during every trip, so the COLREGs compliance is written in
    the same pass.
    """
    evaluate = engine()
    from crowd_nav import measure_compliance
    cfg = settings(exp)
    maps = [m for m in MAP_KEYS if exp.ships[m]]
    evaluate.TripRunner(cfg)  # a wrong setting should stop us here, not inside every worker
    rules = measure_compliance.rule_settings(cfg)

    trips = [(m, ships, seed) for m in maps for ships in exp.ships[m] for seed in range(args.trips)]
    print('{}: policy {}, model {}, {} trips on {} processes{}'.format(
        exp.name, exp.policy, exp.model, len(trips), min(args.jobs, len(trips)),
        ', settings: ' + ' '.join(exp.extra) if exp.extra else ''))
    started, clock, progress = datetime.datetime.now(), time.time(), Progress()
    sailed = evaluate.run_trips(
        cfg, [(evaluate.MAPS[m], ships, seed) for m, ships, seed in trips], jobs=args.jobs,
        task=functools.partial(measure_compliance.measure_trip, rules=rules),
        on_result=lambda done, total, result: progress(done, total, result[0]))
    seconds = time.time() - clock
    rows = [row for row, _, _ in sailed]

    os.makedirs(args.results, exist_ok=True)
    for m in maps:
        with open(csv_path(args.results, exp.name, m), 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=evaluate.CSV_FIELDS)
            writer.writeheader()
            writer.writerows(row for (key, _, _), row in zip(trips, rows) if key == m)
        print('wrote {}'.format(shown(csv_path(args.results, exp.name, m))))
    measured = [(m, record, violations) for (m, _, _), (row, record, violations) in zip(trips, sailed)
                if row['outcome'] != 'scenario_failed']
    measure_compliance.save(exp.name, args.results, measured)
    with open(os.path.join(args.results, exp.name + '.json'), 'w') as fh:
        json.dump(run_record(exp, cfg, args, rules, started, seconds), fh, indent=2)
    print('"{}" finished in {}\n'.format(exp.name, minutes(seconds)))
    outcome_table(rows)
    measure_compliance.report(exp.name, measured)
    return rows


def cmd_run(args):
    chosen = [experiment(name) for name in args.names]
    if args.missing:
        chosen += [exp for exp in load_table().values()
                   if exp not in chosen and not has_results(args.results, exp)]
        if not chosen:
            print('Every experiment of the table has results already.')
            return
    if not chosen:
        raise Problem('Name the experiments to run, or add --missing for every one that has no '
                      'results yet.')
    # stop before the first trip, not hours in, when one of them would replace results
    done = [exp.name for exp in chosen if has_results(args.results, exp)]
    if done and not args.again:
        raise Problem('Results for {} already exist in {}.\nRunning again would replace them. '
                      'Add --again if that is what you want.'.format(
                          ', '.join('"{}"'.format(name) for name in done), shown(args.results)))
    for number, exp in enumerate(chosen, 1):
        if len(chosen) > 1:
            print('--- {} of {} ---'.format(number, len(chosen)))
        run_experiment(exp, args)
        print()


# -- video, demo ------------------------------------------------------------------------

def cmd_video(args):
    evaluate = engine()
    from crowd_nav import measure_compliance, render_samples
    exp = experiment(args.name)
    cfg = settings(exp)
    picks = render_samples.choose(exp.name, args.results, maps=args.maps, types=args.types,
                                  per_type=args.per_type, ships=args.ships, trips=args.trips)
    if args.dry_run or not picks:
        return
    out_dir = os.path.join(args.results, 'videos', exp.name)
    written, mismatched = render_samples.film_trips(
        single_runner(evaluate, cfg), exp.name, picks, out_dir, measure_compliance.rule_settings(cfg))
    print('\n{} video(s) in {}'.format(len(written), shown(out_dir)))
    if mismatched:
        print('{} trip(s) did not replay identically and were skipped: {}'.format(
            len(mismatched), mismatched))


def single_runner(evaluate, cfg):
    import torch
    torch.set_num_threads(1)  # the same as a batch worker, so replays are bit-identical
    return evaluate.TripRunner(cfg)


def cmd_demo(args):
    """One trip, filmed. Needs no experiment to have been run."""
    evaluate = engine()
    from crowd_nav import measure_compliance, render_samples
    names = [args.against, args.experiment] if args.compare else [args.experiment]
    trip = (evaluate.MAPS[args.map], args.ships, args.seed)
    panels = []
    for name in names:
        exp = experiment(name)
        cfg = settings(exp)
        label = LABELS.get(exp.policy, exp.policy)
        print('Sailing {} with {} ships, seed {}, under {} ({}) ...'.format(
            args.map.upper(), args.ships, args.seed, label, exp.name), flush=True)
        runner = single_runner(evaluate, cfg)
        row, watch = measure_compliance.watch_trip(runner, trip, measure_compliance.rule_settings(cfg))
        if row['outcome'] == 'scenario_failed':
            raise Problem('No trip can be laid out on {} for seed {}. Try another seed.'.format(
                args.map.upper(), args.seed))
        summary = watch.tracker.summary()
        if row['n_dynamic'] != args.ships:
            print('  the map only had room for {} of the {} ships'.format(row['n_dynamic'], args.ships))
        print('  {} after {:.2f} s, {} encounters, {} of them sailed by the rules'.format(
            row['outcome'], row['sim_time'], summary['encounters'], summary['compliant']))
        panels.append((runner.env(trip[0]),
                       render_samples.caption(label, args.map, args.ships, args.seed, row, watch),
                       watch.frames))
    out = args.out or os.path.join(args.results, 'videos', 'demo', '{}_{}_{}ships_seed{}.mp4'.format(
        'compare' if args.compare else args.experiment, args.map, args.ships, args.seed))
    print('Filming ...', flush=True)
    print('\nVideo: {}'.format(os.path.abspath(render_samples.film(panels, out))))


# -- colregs ----------------------------------------------------------------------------

def cmd_colregs(args):
    """The report. Runs the experiments that have no results, then prints and saves it."""
    engine()
    from crowd_nav import colregs_report, measure_compliance
    for name in dict.fromkeys(name for pair in COLREGS for name in pair):   # each once
        exp = experiment(name)
        if args.again or not has_results(args.results, exp):
            run_experiment(exp, args)
            print()
        sailed = sum(1 for m in MAP_KEYS for r in read_rows(csv_path(args.results, name, m))
                     if r['outcome'] != 'scenario_failed')
        checked = sum(measure_compliance.checked(args.results, name, m) for m in MAP_KEYS)
        if checked < sailed:   # results from before the rule check: replay every trip
            cfg = settings(exp)
            measured, mismatched = measure_compliance.measure(
                name, cfg, args.results, measure_compliance.rule_settings(cfg), sample=0,
                jobs=args.jobs)
            measure_compliance.report(name, measured, mismatched)
            measure_compliance.save(name, args.results, measured)
            print()

    rules = measure_compliance.rule_settings(settings(experiment(COLREGS[0][1])))
    text = colregs_report.report(
        COLREGS, args.results,
        thresholds='A ship counts as a risk of collision within {max_range:g} m and {max_tcpa:g} s; '
                   'passing {safe_cpa:g} m clear is safe.'.format(**rules))
    path = os.path.join(args.results, 'colregs_report.md')
    with open(path, 'w') as fh:
        fh.write(text)
    print(text)
    print('Saved as {}'.format(shown(path)))


# -- command line -----------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(prog='python -m crowd_nav.exp', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', metavar='COMMAND')
    jobs = max(1, (os.cpu_count() or 2) - 1)

    def command(name, function, summary):
        sub = commands.add_parser(name, help=summary, description=summary)
        sub.set_defaults(function=function)
        sub.add_argument('--results', default=RESULTS, metavar='FOLDER',
                         help='where results are read and written (default: crowd_nav/results-v2)')
        return sub

    sub = command('demo', cmd_demo, 'Film one trip. Needs no experiment to have been run.')
    sub.add_argument('--compare', action='store_true',
                     help='film the same trip twice, side by side: without the rules and with them')
    sub.add_argument('--map', default=DEMO['map'], choices=MAP_KEYS, help='(default: %(default)s)')
    sub.add_argument('--ships', type=int, default=DEMO['ships'], help='other ships (default: %(default)s)')
    sub.add_argument('--seed', type=int, default=DEMO['seed'],
                     help='which trip: the seed fixes start, goal and traffic (default: %(default)s)')
    sub.add_argument('--experiment', default=DEMO['experiment'],
                     help='whose settings to sail under (default: %(default)s)')
    sub.add_argument('--against', default=DEMO['against'],
                     help='the experiment on the left of --compare (default: %(default)s)')
    sub.add_argument('--out', default=None, metavar='FILE',
                     help='video file to write (default: in the results folder, under videos/demo)')

    sub = command('colregs', cmd_colregs,
                  'The report: collision rate, success rate and give-way compliance of SARL '
                  'without and with the rule filter. Runs what is missing, then prints and '
                  'saves it.')
    sub.add_argument('--again', action='store_true',
                     help='run the experiments again instead of using the results on file')
    sub.add_argument('--jobs', type=int, default=jobs, help='trips sailed at once (default: %(default)s)')
    sub.add_argument('--trips', type=int, default=TRIPS,
                     help='trips per map per ship count for experiments that have to be run '
                          '(default: %(default)s)')

    sub = command('run', cmd_run, 'Run experiments: one CSV per map, with the rules checked on '
                                  'every trip.')
    sub.add_argument('names', nargs='*', metavar='NAME', help='the experiments to run, one after another')
    sub.add_argument('--missing', action='store_true',
                     help='run every experiment of the table that has no results yet')
    sub.add_argument('--again', action='store_true', help='replace results that already exist')
    sub.add_argument('--jobs', type=int, default=jobs, help='trips sailed at once (default: %(default)s)')
    sub.add_argument('--trips', type=int, default=TRIPS,
                     help='trips per map per ship count, seeds 0..N-1 (default: %(default)s)')

    sub = command('video', cmd_video, 'Film trips of an experiment.')
    sub.add_argument('name')
    sub.add_argument('trips', nargs='*', metavar='MAP:SHIPS:SEED',
                     help='exactly these trips, e.g. ht:8:60 sf:8:36; leave out for one of each kind')
    from_csv = sub.add_argument_group('when no trip is named (the experiment must have been run)')
    from_csv.add_argument('--types', nargs='+', default=None, metavar='KIND',
                          help='kinds of trip to film (default: all, see crowd_nav/render_samples.py)')
    from_csv.add_argument('--per_type', type=int, default=1, help='trips per kind per map')
    from_csv.add_argument('--maps', nargs='+', default=list(MAP_KEYS), choices=MAP_KEYS)
    from_csv.add_argument('--ships', type=int, nargs='+', default=None, help='only these ship counts')
    from_csv.add_argument('--dry_run', action='store_true', help='list the chosen trips, film nothing')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1
    if getattr(args, 'types', None) is None and args.command == 'video':
        from crowd_nav.render_samples import TYPES   # needs matplotlib, hence not at the top
        args.types = TYPES
    try:
        return args.function(args) or 0
    except KeyboardInterrupt:
        print('\nStopped. Results are written only when a command finishes; videos already made are kept.')
        return 130
    except Exception as error:
        if isinstance(error, tuple(USER_ERRORS)):
            print('error: {}'.format(error), file=sys.stderr)
            return 1
        raise


if __name__ == '__main__':
    sys.exit(main())
