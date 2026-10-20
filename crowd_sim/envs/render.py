"""Frame-by-frame playback of a finished episode, plus video export.

This module deliberately avoids ``matplotlib.pyplot``: the Qt GUI embeds the very same
artists in an existing axes, and pyplot's global figure manager would fight both the
embedded canvas and the offscreen Agg figure used for export. ``CrowdSim_test.render``
still owns the pyplot figures it shows; it just delegates the scene to
:class:`EpisodeAnimation`.

The static land is drawn as one ``PatchCollection`` built from ``env.land_rects``. The
previous per-cell ``Rectangle`` loop added up to a few thousand separate artists, which is
what made an embedded canvas redraw too slowly to drag.
"""
import logging
import math
import os
import shutil

import numpy as np
from matplotlib import animation, rcParams
from matplotlib import lines as mlines
from matplotlib import patches
from matplotlib.collections import PatchCollection

ROBOT_COLOR = 'yellow'
START_COLOR = 'blue'
GOAL_COLOR = 'red'
OUTER_COLOR = 'm'
DETECT_COLOR = 'orange'
ARROW_COLOR = 'red'
ARROW_STYLE = patches.ArrowStyle('->', head_length=4, head_width=2)
LAND_COLOR = 'gray'
LAND_ALPHA = 0.5

# Raw per-frame velocity direction jitters when ORCA makes small avoidance corrections
# (worse near the destination, where speed -> 0 and direction noise dominates). Low-pass
# filter the heading on the unit circle (to avoid +-pi wrap-around artifacts) and hold
# the last heading while nearly stopped.
HEADING_SMOOTHING_ALPHA = 0.3
MIN_SPEED_FOR_HEADING = 0.05

DEFAULT_DETECT_RADIUS = 5.0


def find_ffmpeg():
    """
    Locate an ffmpeg binary. PATH first, then the one shipped with imageio-ffmpeg,
    then the usual system location. Hard coding /usr/bin/ffmpeg breaks on any machine
    that keeps it somewhere else, a conda environment for instance.
    """
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is not None:
        return ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return '/usr/bin/ffmpeg' if os.path.exists('/usr/bin/ffmpeg') else None


def land_rectangles(env):
    """The land geometry the simulator actually collides with, as (x0, y0, x1, y1) rows."""
    rects = getattr(env, 'land_rects', None)
    if rects is None:
        return np.empty((0, 4))
    return np.asarray(rects, dtype=float).reshape(-1, 4)


class EpisodeAnimation(object):
    """The mutable artists of one finished episode, drawn into an axes the caller owns.

    The caller keeps the axes (a Qt canvas in the GUI, a pyplot figure in ``render``) and
    drives the scene either through :meth:`animate` or by calling :meth:`update` itself.
    """

    def __init__(self, env, ax, draw_land=True, detect_radius=DEFAULT_DETECT_RADIUS):
        if not env.states:
            raise ValueError('EpisodeAnimation needs a finished episode in env.states')
        self.env = env
        self.ax = ax
        self.detect_radius = float(detect_radius)
        self.frame = 0

        ax.tick_params(labelsize=16)
        ax.set_xlim(env.env_x_min, env.env_x_max)
        ax.set_ylim(env.env_y_min, env.env_y_max)
        ax.set_xlabel('x(m)', fontsize=16)
        ax.set_ylabel('y(m)', fontsize=16)

        self.land = None
        if draw_land:
            self.land = self._draw_land(ax)

        # the global path the ship was told to follow
        path_x, path_y = zip(*list(env.global_path))
        ax.plot(path_x, path_y, 'k--', label='Global Path', linewidth=1)

        # start and end points of the ship
        start = mlines.Line2D([env.robot_start[0]], [env.robot_start[1]], color=START_COLOR,
                              marker='s', linestyle='None', markersize=6, label='Start')
        goal = mlines.Line2D([env.robot_end[0]], [env.robot_end[1]], color=GOAL_COLOR,
                             marker='*', linestyle='None', markersize=15, label='Goal')
        ax.add_artist(start)
        ax.add_artist(goal)

        # the ship, the local goal it is currently steering at, and its two rings
        self.robot_positions = [state[0].position for state in env.states]
        self.goal_positions = [(state[0].gx, state[0].gy) for state in env.states]
        self.outer_radius = [math.dist((state[0].px, state[0].py), (state[0].gx, state[0].gy))
                             for state in env.states]

        self.robot = patches.Circle(self.robot_positions[0], env.robot.radius, fill=True,
                                    color=ROBOT_COLOR)
        ax.add_artist(self.robot)
        self.goal_local = patches.Circle((env.robot.gx, env.robot.gy), env.robot.radius,
                                         fill=True, color=OUTER_COLOR)
        ax.add_artist(self.goal_local)
        self.robot_outer = patches.Circle(self.robot_positions[0], env.robot.radius, fill=False,
                                          color=OUTER_COLOR, linestyle='--', linewidth=1)
        ax.add_artist(self.robot_outer)
        self.robot_detector = patches.Circle(self.robot_positions[0], self.detect_radius,
                                             fill=False, color=DETECT_COLOR, linestyle='--',
                                             linewidth=1)
        ax.add_artist(self.robot_detector)
        ax.legend([self.robot, goal], ['Robot', 'Goal'], fontsize=16, loc='upper right')

        # the other ships
        self.human_positions = [[state[1][j].position for j in range(len(env.humans))]
                                for state in env.states]
        self.humans = [patches.Circle(self.human_positions[0][i], env.humans[i].radius,
                                      fill=True, color='b')
                       for i in range(len(env.humans))]
        for human in self.humans:
            ax.add_artist(human)

        # ax.text() already registers the artist; adding it again would put the same Text in
        # the axes twice (the original code did exactly that through plt.text + add_artist).
        self.time_text = ax.text(env.env_x_min + 1, env.env_y_max - 3,
                                 'Time: {}'.format(0), fontsize=16)

        self.orientations = self._compute_orientations()
        self.arrow_segments = [orientation[0] for orientation in self.orientations]
        self.arrows = [self._arrow(segment) for segment in self.arrow_segments]
        for arrow in self.arrows:
            self.ax.add_artist(arrow)

    def _draw_land(self, ax):
        rects = land_rectangles(self.env)
        if not len(rects):
            return None
        cells = [patches.Rectangle((x0, y0), x1 - x0, y1 - y0)
                 for x0, y0, x1, y1 in rects]
        collection = PatchCollection(cells, facecolor=LAND_COLOR, alpha=LAND_ALPHA,
                                     edgecolor='none')
        ax.add_collection(collection)
        return collection

    def _arrow(self, orientation):
        return patches.FancyArrowPatch(*orientation, color=ARROW_COLOR, arrowstyle=ARROW_STYLE)

    def _compute_orientations(self):
        """Per-agent (tail, head) heading segments for every frame, in agent order.

        Index 0 is the ship, the rest follow ``env.humans``. Returned as
        ``orientations[agent][frame]``.
        """
        radius = self.env.robot.radius
        if self.env.robot.kinematics == 'unicycle':
            orientation = [((state[0].px, state[0].py),
                            (state[0].px + radius * np.cos(state[0].theta),
                             state[0].py + radius * np.sin(state[0].theta)))
                           for state in self.env.states]
            return [orientation]

        orientations = []
        for i in range(self.env.human_num + 1):
            orientation = []
            smoothed_theta = None
            for state in self.env.states:
                agent_state = state[0] if i == 0 else state[1][i - 1]
                speed = np.hypot(agent_state.vx, agent_state.vy)
                if speed > MIN_SPEED_FOR_HEADING:
                    raw_theta = np.arctan2(agent_state.vy, agent_state.vx)
                    if smoothed_theta is None:
                        smoothed_theta = raw_theta
                    else:
                        blended = (1 - HEADING_SMOOTHING_ALPHA) * np.array(
                            [np.cos(smoothed_theta), np.sin(smoothed_theta)]) + \
                            HEADING_SMOOTHING_ALPHA * np.array([np.cos(raw_theta),
                                                                np.sin(raw_theta)])
                        smoothed_theta = np.arctan2(blended[1], blended[0])
                elif smoothed_theta is None:
                    smoothed_theta = 0.0
                orientation.append(((agent_state.px, agent_state.py),
                                    (agent_state.px + radius * np.cos(smoothed_theta),
                                     agent_state.py + radius * np.sin(smoothed_theta))))
            orientations.append(orientation)
        return orientations

    @property
    def frames(self):
        return len(self.env.states)

    def update(self, k):
        """Move every artist to frame ``k``. Safe to call repeatedly and out of order."""
        self.frame = k
        self.robot.center = self.robot_positions[k]
        self.robot_outer.center = self.robot_positions[k]
        self.robot_outer.radius = self.outer_radius[k]
        self.robot_detector.center = self.robot_positions[k]
        self.robot_detector.radius = self.detect_radius
        self.goal_local.center = self.goal_positions[k]
        for i, human in enumerate(self.humans):
            human.center = self.human_positions[k][i]
        # The heading arrows belong to the frame, not to any single ship: rebuilding them
        # inside the human loop above rebuilt every arrow once per human and left them
        # frozen at frame 0 when the episode had no other ships at all.
        for arrow in self.arrows:
            arrow.remove()
        self.arrow_segments = [orientation[k] for orientation in self.orientations]
        self.arrows = [self._arrow(segment) for segment in self.arrow_segments]
        for arrow in self.arrows:
            self.ax.add_artist(arrow)
        self.time_text.set_text('Time: {:.2f}'.format(k * self.env.time_step))

    def animate(self, fig, interval_ms, repeat=False):
        """A ``FuncAnimation`` over this scene's frames, for pyplot playback and saving."""
        anim = animation.FuncAnimation(fig, self.update, frames=self.frames,
                                       interval=interval_ms, repeat=repeat)
        anim.running = True
        return anim


def save_animation(anim, output_file, fps=8, progress=None):
    """Write ``anim`` to ``output_file``; fall back to a gif without ffmpeg.

    Returns the path actually written, which differs from ``output_file`` on the gif
    fallback. ``progress`` is matplotlib's ``progress_callback``: ``progress(i, n)``.
    """
    ffmpeg = find_ffmpeg()
    if ffmpeg is not None:
        rcParams['animation.ffmpeg_path'] = ffmpeg
    if animation.writers.is_available('ffmpeg'):
        writer = animation.writers['ffmpeg'](fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    else:
        # no ffmpeg on this machine, fall back to a gif so the run is not lost
        output_file = os.path.splitext(output_file)[0] + '.gif'
        logging.warning('ffmpeg not found, writing %s instead. Install it with '
                        '"conda install -c conda-forge ffmpeg" or "pip install imageio-ffmpeg"',
                        output_file)
        writer = animation.PillowWriter(fps=fps)
    anim.save(output_file, writer=writer, progress_callback=progress)
    logging.info('Video written to %s', output_file)
    return output_file
