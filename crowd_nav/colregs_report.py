"""The COLREGs result: collision rate, success rate and give-way compliance of SARL
without and with the rule filter, for each map and number of ships.

    python -m crowd_nav.exp colregs

It only reads what `exp run` wrote in the results folder: NAME_<map>.csv (one row per
trip), compliance/NAME_<map>.csv (the rule check) and NAME.json (where the run came from).
"""
import csv
import datetime
import json
import os
import textwrap

MAPS = ('sf', 'ht', 'sg')


def short(map_name):
    """'land_shapes_ht_crop', as the CSVs spell a map, is 'ht'."""
    return map_name.replace('land_shapes_', '').replace('_crop', '')


def read(folder, name, ships_field):
    """{(map, ships, seed): row} of an experiment's CSVs in ``folder``."""
    rows = {}
    for m in MAPS:
        path = os.path.join(folder, '{}_{}.csv'.format(name, m))
        if not os.path.exists(path):
            continue
        with open(path, newline='') as fh:
            for r in csv.DictReader(fh):
                rows[(short(r['map']), int(r[ships_field]), int(r['seed']))] = r
    return rows


def number(row, field):
    return int(row.get(field) or 0)


def share(k, n):
    """'12.0% (36 of 300)'."""
    return '-' if not n else '{:.1f}% ({} of {})'.format(100 * k / n, k, n)


def table(rows):
    """A Markdown table with its columns lined up, so it also reads well in a terminal."""
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for n, row in enumerate(rows):
        lines.append('| ' + ' | '.join(cell.ljust(w) for cell, w in zip(row, widths)) + ' |')
        if n == 0:
            lines.append('|' + '|'.join('-' * (w + 2) for w in widths) + '|')
    return lines


def prose(text, bullet=False):
    """A paragraph or a bullet, folded to the width of the tables."""
    return textwrap.wrap(text, width=100, initial_indent='- ' if bullet else '',
                         subsequent_indent='  ' if bullet else '')


def source(results, name):
    """One line on where an experiment's numbers came from."""
    path = os.path.join(results, name + '.json')
    if not os.path.exists(path):
        return '`{}`: run before runs were recorded, commit not known'.format(name)
    with open(path) as fh:
        record = json.load(fh)
    return '`{}`: run {}, commit {}{}, settings `{}`'.format(
        name, record.get('started', '?'), record.get('commit') or '?',
        ' plus uncommitted changes' if record.get('local_changes') else '',
        ' '.join(record.get('settings') or []) or 'none')


def cells(pair, results):
    """The rows of one pair of experiments, over the trips both of them sailed.

    Returns ([(map, ships, name, trips, collisions, successes, (kept, of))], partial);
    ``partial`` is true when the rule check does not cover every one of those trips.
    """
    plain, ruled = pair
    A, B = read(results, plain, 'requested_n_dynamic'), read(results, ruled, 'requested_n_dynamic')
    folder = os.path.join(results, 'compliance')
    CA, CB = read(folder, plain, 'ships'), read(folder, ruled, 'ships')
    keys = sorted(k for k in set(A) & set(B)
                  if 'scenario_failed' not in (A[k]['outcome'], B[k]['outcome']))
    found, partial = [], False
    for m, ships in sorted({k[:2] for k in keys}, key=lambda c: (MAPS.index(c[0]), c[1])):
        trips = [k for k in keys if k[:2] == (m, ships)]
        checked = [k for k in trips if k in CA and k in CB]
        partial = partial or len(checked) < len(trips)
        for name, outcomes, rules in ((plain, A, CA), (ruled, B, CB)):
            found.append((m, ships, name, len(trips),
                          sum(outcomes[k]['outcome'].startswith('collision') for k in trips),
                          sum(outcomes[k]['outcome'] == 'success' for k in trips),
                          (sum(number(rules[k], 'give_way_compliant') for k in checked),
                           sum(number(rules[k], 'give_way_encounters') for k in checked))))
    return found, partial


def report(pairs, results, thresholds=''):
    """The whole report as Markdown text. ``pairs`` holds (without the rules, with them)."""
    found, partial = [], False
    for pair in pairs:
        more, short_of = cells(pair, results)
        found += [f for f in more if all(f[:3] != g[:3] for g in found)]   # an experiment once per row
        partial = partial or short_of
    lines = ['# COLREGs result: SARL without and with the rule filter', '']
    lines += prose('Made on {} by `python -m crowd_nav.exp colregs`. The two policies of a pair sail '
                   'the same trips (same map, number of ships and seed). The other ships can see '
                   'ours.'.format(datetime.date.today())) + ['']
    for m in MAPS:
        mine = sorted((f for f in found if f[0] == m), key=lambda f: f[1])   # stable: pairs stay together
        if not mine:
            continue
        lines += ['## {}'.format(m.upper()), '']
        lines += table([['ships', 'experiment', 'trips', 'collision rate', 'success rate',
                         'give-way compliance']] +
                       [[str(ships), name, str(trips), share(collisions, trips),
                         share(successes, trips), share(*give_way)]
                        for _, ships, name, trips, collisions, successes, give_way in mine]) + ['']
    if not found:
        lines += ['No trips to compare yet: run the experiments first.', '']
        return '\n'.join(lines) + '\n'

    lines += ['## Reading the table', '']
    notes = ['Collision rate: the share of trips that ended in a collision, with a ship or with land. '
             'Success rate: the share that reached the goal. The rest ran out of time.',
             'Give-way compliance: of the encounters in which our ship had to give way, the share '
             'sailed by the rules, meaning no step of the encounter broke its rule. It counts '
             'encounters, not trips.',
             'With only {} trips per row, a difference of a few points between two rows is within '
             'what other trips would change.'.format(min(f[3] for f in found))]
    if partial:
        notes.append('The rule check covers only some of the trips of at least one row.')
    for note in notes + ([thresholds] if thresholds else []):
        lines += prose(note, True)
    lines += ['', '## Where the numbers come from', '']
    for name in dict.fromkeys(f[2] for f in found):
        lines += prose(source(results, name), True)
    return '\n'.join(lines) + '\n'
