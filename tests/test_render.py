"""Episode playback artists and video export (matplotlib Agg, no GUI)."""
import matplotlib
matplotlib.use('Agg', force=True)

import numpy as np
import pytest
from matplotlib import animation, patches, pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from conftest import FixedPolicy
from crowd_sim.envs.render import EpisodeAnimation, find_ffmpeg, land_rectangles
from crowd_sim.envs.utils.action import ActionXY

DYNAMIC = [dict(px=5.5, py=6.5, gx=8, gy=8), dict(px=6.5, py=4.5, gx=9, gy=9)]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def run_steps(env, steps=3, action=ActionXY(1, 0)):
    """Drive a reset env for `steps`, so env.states holds `steps` frames."""
    env.robot.set_policy(FixedPolicy(action))
    for _ in range(steps):
        ob = env.light_reset('test', 0)
        env.step(env.robot.act(ob))
    return env


def axes():
    fig = Figure(figsize=(7, 7))
    FigureCanvasAgg(fig)
    return fig, fig.add_subplot(111)


def arrow_artists(ax):
    # add_artist() appends to Axes._children, which ax.get_children() exposes; the legacy
    # ax.artists list stays empty on matplotlib >= 3.7.
    return [a for a in ax.get_children() if isinstance(a, patches.FancyArrowPatch)]


def test_update_moves_every_artist_to_the_frame(make_env):
    env = run_steps(make_env(dynamic=DYNAMIC))
    fig, ax = axes()
    scene = EpisodeAnimation(env, ax)
    assert scene.frames == len(env.states) == 3

    scene.update(2)
    assert scene.frame == 2
    assert tuple(scene.robot.center) == env.states[2][0].position
    assert tuple(scene.humans[0].center) == env.states[2][1][0].position
    assert tuple(scene.humans[1].center) == env.states[2][1][1].position
    assert scene.time_text.get_text() == 'Time: 0.50'
    # ax.text() registers the artist itself; a second add_artist() would list it twice
    assert list(ax.texts) == [scene.time_text]


def test_land_comes_from_the_collision_geometry(make_env):
    env = make_env(obstacles=[dict(cx=7.5, cy=5.5, w=1, h=1)])
    run_steps(env)
    assert np.allclose(land_rectangles(env), [[7, 5, 8, 6]])
    fig, ax = axes()
    scene = EpisodeAnimation(env, ax)
    assert scene.land is not None
    assert np.allclose(scene.land.get_paths()[0].vertices.min(axis=0), (7, 5))
    # draw_land=False leaves the axes for the caller (the GUI draws its own mask)
    fig2, ax2 = axes()
    assert EpisodeAnimation(env, ax2, draw_land=False).land is None


def test_arrows_are_rebuilt_once_per_frame(make_env):
    env = run_steps(make_env(dynamic=DYNAMIC))
    assert env.human_num == 2
    fig, ax = axes()
    scene = EpisodeAnimation(env, ax)
    assert len(arrow_artists(ax)) == env.human_num + 1

    scene.update(1)
    scene.update(2)
    assert len(scene.arrows) == len(arrow_artists(ax)) == env.human_num + 1
    for segment, orientations in zip(scene.arrow_segments, scene.orientations):
        assert segment == orientations[2]


def test_arrows_still_update_without_other_ships(make_env):
    """Regression: the arrow rebuild used to sit inside the per-human loop, so an episode
    with no other ships froze its heading arrow at frame 0."""
    env = run_steps(make_env(dynamic=[]))
    assert env.human_num == 0
    fig, ax = axes()
    scene = EpisodeAnimation(env, ax)
    assert len(arrow_artists(ax)) == 1

    scene.update(2)
    assert len(arrow_artists(ax)) == 1
    assert scene.arrow_segments[0][0] == env.states[2][0].position
    assert scene.arrow_segments[0][0] != env.states[0][0].position


@pytest.mark.filterwarnings('ignore:Animation was deleted without rendering')
def test_animate_covers_every_frame(make_env):
    env = run_steps(make_env(dynamic=DYNAMIC))
    fig, ax = axes()
    scene = EpisodeAnimation(env, ax)
    anim = scene.animate(fig, env.time_step * 1000)
    assert anim.running is True
    assert list(anim.new_saved_frame_seq()) == [0, 1, 2]


def test_render_video_falls_back_to_gif(make_env, monkeypatch, tmp_path):
    env = run_steps(make_env(dynamic=DYNAMIC))
    monkeypatch.setattr(animation.writers, 'is_available', lambda *args, **kwargs: False)
    out = tmp_path / 'out.mp4'
    env.render('video', str(out))
    assert not out.exists()
    assert (tmp_path / 'out.gif').exists()


def test_find_ffmpeg_is_still_exported_from_crowd_sim():
    from crowd_sim.envs import crowd_sim
    assert crowd_sim.find_ffmpeg is find_ffmpeg
