"""Scores the actions a ship actually took against the COLREGs rules.

A ``RuleWatch`` looks on during every trip and gives each action to the encounter tracker
of colregs.py. `exp run` keeps it on, so the compliance comes with the results. Plain SARL
is scored the same way, which is what the rule filter is compared with.

One row per trip goes to <results>/compliance/<name>_<map>.csv.
"""
import csv
import functools
import os
from collections import namedtuple

from crowd_nav.evaluate import MAPS, map_key, read_config, run_trips
from crowd_nav.policy import colregs
from crowd_nav.policy.sarl_colregs import rule_thresholds
from crowd_sim.envs.utils.action import ActionXY

KINDS = [colregs.HEAD_ON, colregs.CROSSING, colregs.OVERTAKING, colregs.OVERTAKEN]
# our duty in an encounter; the filter only enforces give way
ROLES = [colregs.GIVE_WAY, colregs.STAND_ON]
FIELDS = (['map', 'seed', 'ships', 'outcome', 'steps', 'encounters', 'compliant_encounters',
           'compliance_rate', 'violating_steps', 'encounter_steps', 'stopped_steps']
          + ['{}_encounters'.format(k) for k in KINDS] + ['{}_compliant'.format(k) for k in KINDS]
          + ['{}_encounters'.format(r) for r in ROLES] + ['{}_compliant'.format(r) for r in ROLES])

# what another ship is to ours in one frame: kind of encounter, our role, rule broken or not
Mark = namedtuple('Mark', 'kind role broken')


def rule_settings(cfg, **given):
    """The thresholds to score with: the experiment's own, unless given explicitly."""
    rules = rule_thresholds(read_config(cfg, 'policy'))
    rules.update({key: value for key, value in given.items() if value is not None})
    return rules


class RuleWatch(object):
    """An ``on_step`` callback that scores every executed action against the rules.

    ``frames[k]`` holds {ship index: Mark} for frame k (the videos colour ships with it);
    ``violations`` lists the steps that broke a rule.
    """

    def __init__(self, env, robot, rules):
        self.env = env
        self.robot = robot
        self.tracker = colregs.EncounterTracker(
            clear_range=rules['clear_range'], safe_cpa=rules['safe_cpa'],
            min_turn=rules['min_turn'], max_range=rules['max_range'], max_tcpa=rules['max_tcpa'])
        self.frames = []
        self.violations = []

    def __call__(self, steps, _max_steps):
        own, ships = self.env.states[-1][0], self.env.states[-1][1]  # the state the action was chosen in
        action = ActionXY(self.robot.vx, self.robot.vy)              # holonomic: velocity is the action
        marks = {}
        for key, encounter in self.tracker.update(own, ships, action=action).items():
            broken = self.tracker.breaks(action, encounter, own, ships[key])
            if broken:
                self.violations.append((steps, encounter.kind, encounter.role))
            marks[key] = Mark(encounter.kind, encounter.role, broken)
        self.frames.append(marks)

    def record(self, row):
        """The compliance row of the trip whose metrics row is ``row``."""
        summary = self.tracker.summary()
        record = {
            'map': row['map'], 'seed': row['seed'], 'ships': row['requested_n_dynamic'],
            'outcome': row['outcome'], 'steps': row['steps'],
            'encounters': summary['encounters'], 'compliant_encounters': summary['compliant'],
            'compliance_rate': ('' if summary['compliance_rate'] is None
                                else round(summary['compliance_rate'], 4)),
            'violating_steps': summary['violating_steps'], 'encounter_steps': summary['steps'],
            'stopped_steps': summary['stopped_steps'],
        }
        for table, names in (('by_kind', KINDS), ('by_role', ROLES)):
            for name in names:
                entry = summary[table].get(name, {'encounters': 0, 'compliant': 0})
                record['{}_encounters'.format(name)] = entry['encounters']
                record['{}_compliant'.format(name)] = entry['compliant']
        return record


def watch_trip(runner, trip, rules):
    """Sail one trip under the rule watch; returns (metrics row, the watch)."""
    map_path, ships, seed = trip
    env, robot = runner.stage(map_path)[:2]
    watch = RuleWatch(env, robot, rules)
    return runner.run(map_path, ships, seed, on_step=watch), watch


def measure_trip(runner, trip, rules):
    """The task ``run_trips`` gives its workers: (metrics row, compliance row, violations)."""
    row, watch = watch_trip(runner, trip, rules)
    return row, watch.record(row), watch.violations


def spread(rows, sample):
    """Take `sample` rows spread evenly over the experiment, so every density is represented."""
    rows = sorted(rows, key=lambda r: (int(r['requested_n_dynamic']), int(r['seed'])))
    if not sample or sample >= len(rows):
        return rows
    step = len(rows) / float(sample)
    return [rows[int(i * step)] for i in range(sample)]


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


def measure(name, cfg, results, rules, maps=('sf', 'ht', 'sg'), sample=25, ships=None, jobs=None):
    """Replay trips of an experiment under the rule watch (``sample`` of them, 0 for all).

    Returns (measured, mismatched): a list of (map key, compliance row, violations), and
    how many replays did not match their CSV row. Nothing is written; see :func:`save`.
    """
    chosen = []
    for m in maps:
        csv_path = os.path.join(results, '{}_{}.csv'.format(name, m))
        if not os.path.exists(csv_path):
            print('{}: {} not found, skipped'.format(m.upper(), csv_path))
            continue
        with open(csv_path, newline='') as fh:
            rows = [r for r in csv.DictReader(fh) if r['outcome'] != 'scenario_failed']
        if ships:
            rows = [r for r in rows if int(r['requested_n_dynamic']) in ships]
        rows = spread(rows, sample)
        print('{}: replaying {} trips'.format(m.upper(), len(rows)))
        chosen += [(m, r) for r in rows]
    if not chosen:
        return [], 0

    def show(done, total, result):
        _, record, violations = result
        print('  [{:3d}/{}] {} seed {:3d}, {:2d} ships: {:2d} encounters, {:2d} violated, {}'.format(
            done, total, map_key(record['map']).upper(), record['seed'], record['ships'],
            record['encounters'], record['encounters'] - record['compliant_encounters'],
            'clean' if not violations else times(violations)), flush=True)

    trips = [(MAPS[m], int(r['requested_n_dynamic']), int(r['seed'])) for m, r in chosen]
    replays = run_trips(cfg, trips, jobs=jobs, on_result=show,
                        task=functools.partial(measure_trip, rules=rules))
    measured, mismatched = [], 0
    for (m, stored), (row, record, violations) in zip(chosen, replays):
        mismatched += not (row['outcome'] == stored['outcome']
                           and str(row['steps']) == stored['steps'])
        measured.append((m, record, violations))
    return measured, mismatched


def compliance_path(results, name, m):
    return os.path.join(results, 'compliance', '{}_{}.csv'.format(name, m))


def checked(results, name, m):
    """How many trips of one map the rule check on file covers (0 when there is none)."""
    path = compliance_path(results, name, m)
    if not os.path.exists(path):
        return 0
    with open(path, newline='') as fh:
        return sum(1 for _ in csv.DictReader(fh))


def save(name, results, measured):
    """Write <results>/compliance/<name>_<map>.csv from (map key, compliance row, ...) items."""
    os.makedirs(os.path.join(results, 'compliance'), exist_ok=True)
    for m in sorted({item[0] for item in measured}):
        with open(compliance_path(results, name, m), 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(item[1] for item in measured if item[0] == m)


def totals(records):
    """Encounters and how many were complied with, overall and per kind of encounter."""
    total = lambda field: sum(int(r[field] or 0) for r in records)
    return {
        'trips': len(records),
        'trips_with_encounters': sum(1 for r in records if int(r['encounters'] or 0)),
        'encounters': total('encounters'), 'compliant': total('compliant_encounters'),
        'encounter_steps': total('encounter_steps'), 'violating_steps': total('violating_steps'),
        'stopped_steps': total('stopped_steps'),
        'by_kind': {kind: (total('{}_encounters'.format(kind)), total('{}_compliant'.format(kind)))
                    for kind in KINDS},
        'by_role': {role: (total('{}_encounters'.format(role)), total('{}_compliant'.format(role)))
                    for role in ROLES},
    }


def report(name, measured, mismatched=0):
    """Print the totals of a rule check."""
    if not measured:
        return
    t = totals([record for _, record, _ in measured])
    print('\n=== {} : COLREGs compliance over {} trips ==='.format(name, t['trips']))
    print('trips with at least one encounter: {} of {}'.format(t['trips_with_encounters'], t['trips']))
    print('encounters: {}  compliant: {}  compliance rate: {}'.format(
        t['encounters'], t['compliant'],
        '{:.1%}'.format(t['compliant'] / t['encounters']) if t['encounters'] else 'n/a (no encounters)'))
    print('steps inside an encounter: {}  violating: {}  stopped: {}'.format(
        t['encounter_steps'], t['violating_steps'], t['stopped_steps']))
    for title, table, names in (('rule', 'by_kind', KINDS), ('our duty', 'by_role', ROLES)):
        print('\n{:12s} {:>11s} {:>10s} {:>7s}'.format(title, 'encounters', 'compliant', 'rate'))
        for name in names:
            n, ok = t[table][name]
            print('{:12s} {:11d} {:10d} {:>7s}'.format(name, n, ok, '{:.0%}'.format(ok / n) if n else '-'))
    if mismatched:
        print('WARNING: {} replay(s) did not match the CSV, so the settings in the experiments '
              'table are not the ones the run used.'.format(mismatched))
