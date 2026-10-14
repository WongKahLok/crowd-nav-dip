"""Videos of trips: the trips asked for, or one of each kind of outcome of an experiment.

    python -m crowd_nav.exp video colregs ht:8:60 sf:8:36            # these trips (map:ships:seed)
    python -m crowd_nav.exp video colregs                            # one trip of each kind
    python -m crowd_nav.exp video colregs --types collision_dynamic timeout --per_type 3
    python -m crowd_nav.exp video colregs --dry_run                  # list the picks only

A trip replays exactly (the seed fixes the scenario and the policies are deterministic).
When the experiment has been run, the replay is checked against its CSV row before it is
filmed, so a video shows the trip that was measured.

The scene is marked with the rules: a ship we must give way to is ringed in red, a ship we
stand on for in green, the rule in force is written under the clock, and our ship is
ringed in red while it breaks one.

Kinds of trip (--types):
    success_typical    success whose duration is closest to the median success
    success_closest    success with the smallest gap to a moving ship (near miss)
    longest_detour     success with the largest travelled / planned distance
    most_fallback      trip where the emergency fallback acted most (SARL only)
    collision_dynamic  hit a moving ship
    collision_static   hit land or the map edge
    collision_both     hit both within one step
    timeout            ran out of time
"""
import csv
import os
import statistics

from matplotlib import animation, patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from crowd_nav.evaluate import MAPS, SettingsError
from crowd_nav.measure_compliance import watch_trip
from crowd_nav.policy import colregs
from crowd_sim.envs.render import ROBOT_COLOR, EpisodeAnimation, save_animation

TYPES = ['success_typical', 'success_closest', 'longest_detour', 'most_fallback',
         'collision_dynamic', 'collision_static', 'collision_both', 'timeout']

GIVE_WAY_COLOR = '#d62728'   # we have to keep out of this ship's way
STAND_ON_COLOR = '#2ca02c'   # this ship has to keep out of ours
TRAIL_COLOR = '#b8860b'
RING_RADIUS = 1.2            # ships are 0.3 m on a 50 m map: a ring makes the marked ones findable
RULES = {(colregs.HEAD_ON, colregs.GIVE_WAY): 'Head-on, Rule 14: alter to starboard',
         (colregs.CROSSING, colregs.GIVE_WAY): 'Crossing, Rule 15: give way',
         (colregs.CROSSING, colregs.STAND_ON): 'Crossing, Rule 17: stand on',
         (colregs.OVERTAKING, colregs.GIVE_WAY): 'Overtaking, Rule 13: keep clear',
         (colregs.OVERTAKEN, colregs.STAND_ON): 'Being overtaken, Rule 17: stand on'}


def num(row, field):
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError):
        return None


def pick(rows, kind, n, taken):
    """Up to n trips of one kind, skipping trips another kind already took."""
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


def parse_trip(text):
    """'ht:8:60' -> ('ht', 8, 60)."""
    parts = text.split(':')
    try:
        m, ships, seed = parts[0], int(parts[1]), int(parts[2])
        if len(parts) != 3 or m not in MAPS or ships < 0:
            raise ValueError
    except (IndexError, ValueError):
        raise SettingsError('Cannot read the trip "{}": write it as map:ships:seed, for example '
                            'ht:8:60 (maps: {})'.format(text, ', '.join(sorted(MAPS))))
    return m, ships, seed


def choose(name, results, maps=('sf', 'ht'), types=TYPES, per_type=1, ships=None, trips=None):
    """The trips to film, as (kind, map key, ships, seed, CSV row or None)."""
    wanted = [parse_trip(t) for t in trips or ()]
    picks = []
    for m in maps:
        csv_path = os.path.join(results, '{}_{}.csv'.format(name, m))
        rows = []
        if os.path.exists(csv_path):
            with open(csv_path, newline='') as fh:
                rows = [r for r in csv.DictReader(fh) if r['outcome'] != 'scenario_failed']
        chosen = []
        if wanted:
            for _, n, seed in (t for t in wanted if t[0] == m):
                match = [r for r in rows if int(r['requested_n_dynamic']) == n and int(r['seed']) == seed]
                chosen.append(('episode', m, n, seed, match[0] if match else None))
        elif not rows:
            print('{}: no results in {} yet. Run the experiment first, or name the trips to '
                  'film.'.format(m.upper(), csv_path))
        else:
            if ships:
                rows = [r for r in rows if int(r['requested_n_dynamic']) in ships]
            taken = set()
            print('{}: {} trips in {}'.format(m.upper(), len(rows), csv_path))
            for kind in types:
                found = pick(rows, kind, per_type, taken)
                if not found:
                    print('  {:18s} none in this experiment'.format(kind))
                chosen += [(kind, m, int(r['requested_n_dynamic']), int(r['seed']), r) for r in found]
        for kind, _, n, seed, stored in chosen:
            print('  {:18s} {} {:2d} ships, seed {:3d}: {}'.format(
                kind, m.upper(), n, seed,
                '{} after {} steps'.format(stored['outcome'], stored['steps']) if stored
                else 'not in the results, filmed without a check'))
        picks += chosen
    return picks


# -- drawing ------------------------------------------------------------------------

class RuleMarks(object):
    """The COLREGs situation drawn over one scene, from the frames of a ``RuleWatch``."""

    def __init__(self, scene, frames):
        self.scene = scene
        self.frames = frames
        ax = scene.ax
        xs, ys = zip(*scene.robot_positions)
        self.track = (xs, ys)
        self.trail, = ax.plot([], [], color=TRAIL_COLOR, linewidth=1.5, zorder=1)
        self.rings = [patches.Circle((0, 0), RING_RADIUS, fill=False, linewidth=2, visible=False)
                      for _ in scene.humans]
        self.own_ring = patches.Circle((0, 0), RING_RADIUS, fill=False, linewidth=2,
                                       color=GIVE_WAY_COLOR, linestyle=':', visible=False)
        for ring in self.rings + [self.own_ring]:
            ax.add_artist(ring)
        env = scene.env
        self.text = ax.text(env.env_x_min + 1, env.env_y_max - 4.5, '', fontsize=11, va='top',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ring = lambda color: Line2D([], [], marker='o', linestyle='None', markersize=9,
                                    markerfacecolor='none', markeredgewidth=2, color=color)
        ship = lambda color: Line2D([], [], marker='o', linestyle='None', markersize=7, color=color)
        ax.legend([ship(ROBOT_COLOR), ship('b'), ring(GIVE_WAY_COLOR), ring(STAND_ON_COLOR)],
                  ['Our ship', 'Other ships', 'We give way', 'We stand on'],
                  fontsize=10, loc='upper right')

    def update(self, k):
        self.trail.set_data(self.track[0][:k + 1], self.track[1][:k + 1])
        marks = self.frames[k] if k < len(self.frames) else {}
        for i, ring in enumerate(self.rings):
            mark = marks.get(i)
            ring.set_visible(mark is not None)
            if mark is not None:
                ring.center = self.scene.human_positions[k][i]
                ring.set_edgecolor(GIVE_WAY_COLOR if mark.role == colregs.GIVE_WAY else STAND_ON_COLOR)
        broken = any(mark.broken for mark in marks.values())
        self.own_ring.set_visible(broken)
        self.own_ring.center = self.scene.robot_positions[k]
        lines = sorted({RULES[mark.kind, mark.role] + (' - BROKEN' if mark.broken else '')
                        for mark in marks.values()})
        self.text.set_text('\n'.join(lines[:4]))
        self.text.set_color(GIVE_WAY_COLOR if broken else 'black')


def wrap(title, width=72):
    """Keep a long caption inside the frame by breaking it at a separator."""
    if len(title) <= width:
        return title
    parts = title.split(' | ')
    half = max(len(parts) // 2, 1)
    return ' | '.join(parts[:half]) + '\n' + ' | '.join(parts[half:])


def film(panels, path, fps=8):
    """Write one video with a panel per (env, title, frames); returns the path written.

    Each env holds the trip it sailed last. ``frames`` are a ``RuleWatch``'s, or None for
    a plain scene. Panels play side by side on the same clock and a trip that ends first
    holds its last frame, which is what makes two policies comparable at a glance.
    """
    figure = Figure(figsize=(7 * len(panels), 7.3))
    FigureCanvasAgg(figure)
    scenes = []
    for i, (env, title, frames) in enumerate(panels, 1):
        ax = figure.add_subplot(1, len(panels), i)
        scene = EpisodeAnimation(env, ax)
        ax.set_title(wrap(title), fontsize=10)
        scenes.append((scene, RuleMarks(scene, frames) if frames is not None else None))
    figure.tight_layout()

    def update(k):
        for scene, marks in scenes:
            last = min(k, scene.frames - 1)
            scene.update(last)
            if marks is not None:
                marks.update(last)

    anim = animation.FuncAnimation(figure, update, frames=max(scene.frames for scene, _ in scenes),
                                   interval=panels[0][0].time_step * 1000, repeat=False)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return save_animation(anim, path, fps=fps)


def caption(name, m, ships, seed, row, watch):
    """What a video says above its scene: the trip, how it ended, how the rules fared."""
    placed = row['n_dynamic']   # a map that is full takes fewer ships than were asked for
    text = '{} | {} | {} ships | seed {} | {} after {:.2f} s'.format(
        name, m.upper(), ships if placed == ships else '{} of {}'.format(placed, ships), seed,
        row['outcome'], row['sim_time'])
    summary = watch.tracker.summary()
    if summary['encounters']:
        text += ' | {} of {} encounters by the rules'.format(summary['compliant'], summary['encounters'])
    else:
        text += ' | no encounter'
    return text


def film_trips(runner, name, picks, out_dir, rules):
    """Replay and film ``picks`` (from :func:`choose`). Returns (videos written, mismatches)."""
    written, mismatched = [], []
    for kind, m, ships, seed, stored in picks:
        row, watch = watch_trip(runner, (MAPS[m], ships, seed), rules)
        if row['outcome'] == 'scenario_failed':
            print('  ! {} {} ships seed {}: no trip can be laid out for this seed'.format(
                m.upper(), ships, seed))
            continue
        if stored is not None and (row['outcome'] != stored['outcome']
                                   or str(row['steps']) != stored['steps']):
            mismatched.append((m, ships, seed))
            print('  ! {} {} ships seed {}: the replay gave {} after {} steps but the CSV says {} '
                  'after {}. Not filmed: the experiments table no longer holds the settings of '
                  'that run.'.format(m.upper(), ships, seed, row['outcome'], row['steps'],
                                     stored['outcome'], stored['steps']))
            continue
        path = os.path.join(out_dir, '{}_{}_{}ships_seed{}.mp4'.format(m, kind, ships, seed))
        path = film([(runner.env(MAPS[m]), caption(name, m, ships, seed, row, watch), watch.frames)],
                    path)
        written.append(path)
        print('  wrote {}'.format(path))
    return written, mismatched
