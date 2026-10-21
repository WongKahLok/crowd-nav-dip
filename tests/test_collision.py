import math

import pytest

from crowd_sim.envs.utils.collision import circle_clearance, evaluate_motion, static_clearance
from crowd_sim.envs.utils.info import Collision, Danger, Nothing, ReachGoal, Timeout


@pytest.mark.parametrize('start,end,other,expected', [
    ((0, 0), (0, 0), (1, 0), 0),
    ((0, 0), (0, 0), (0.25, 0), -0.75),
    ((-2, 0), (2, 0), (0, 0), -1),
    ((-2, 1), (2, 1), (0, 0), 0),
    ((-2, 2), (2, 2), (0, 0), 1),
])
def test_circle_sweep(start, end, other, expected):
    assert circle_clearance(start, end, 0.5, other, other, 0.5) == pytest.approx(expected)


def test_two_moving_circles():
    assert circle_clearance((-1, 0), (1, 0), 0.3, (1, 0), (-1, 0), 0.3) == pytest.approx(-0.6)


@pytest.mark.parametrize('start,end,radius,expected', [
    ((1.15, 1.15), (1.15, 1.15), 0.3, math.hypot(0.15, 0.15) - 0.3),
    ((-1, 0.5), (2, 0.5), 0.3, -0.3),
    ((-1, 1.5), (2, 1.5), 0.5, 0),
    ((-1, 1.6), (2, 1.6), 0.5, 0.1),
    ((0.5, 0.5), (0.5, 0.5), 0.3, -0.3),
    ((-1, 1), (1, -1), 0.3, -0.3),
])
def test_rectangle_sweep(start, end, radius, expected):
    assert static_clearance(start, end, radius, [(0, 0, 1, 1)]) == pytest.approx(expected)


def test_corner_regression():
    pos = (1.15, 1.15)
    assert math.dist(pos, (0.5, 0.5)) - 0.5 - 0.3 > 0
    assert static_clearance(pos, pos, 0.3, [(0, 0, 1, 1)]) < 0


def test_boundary_and_empty_map():
    assert static_clearance((2, 2), (10, 2), 0.3, bounds=(0, 0, 10, 10)) == pytest.approx(-0.3)
    assert static_clearance((2, 2), (11, 2), 0.3, bounds=(0, 0, 10, 10)) == pytest.approx(-1.3)
    assert static_clearance((2, 2), (9.5, 2), 0.5, bounds=(0, 0, 10, 10)) == 0
    assert math.isinf(static_clearance((0, 0), (1, 1), 0.3))


def assess(**kwargs):
    args = dict(start=(0, 0), end=(1, 0), radius=0.3, humans=[], rects=[], bounds=None,
                local_goal=(10, 0), final_goal=(10, 0), end_time=0.25, time_limit=25,
                time_step=0.25, success_reward=2, collision_penalty=-3,
                discomfort_dist=0.2, discomfort_penalty_factor=0.7)
    args.update(kwargs)
    return evaluate_motion(**args)


@pytest.mark.parametrize('kind', ['dynamic', 'static', 'both'])
def test_collision_source_and_priority(kind):
    humans = [((0.5, 0), (0.5, 0), 0.3)] if kind != 'static' else []
    rects = [(0.4, -0.5, 0.6, 0.5)] if kind != 'dynamic' else []
    reward, done, info = assess(humans=humans, rects=rects, final_goal=(1, 0), end_time=25)
    assert reward == -3 and done and isinstance(info, Collision)
    assert info.collision_kind == kind


def test_goal_timeout_and_local_goal():
    reward, done, info = assess(final_goal=(1, 0), end_time=25)
    assert (reward, done, type(info)) == (2, True, ReachGoal)
    reward, done, info = assess(local_goal=(1, 0))
    assert (reward, done, type(info)) == (2, False, Nothing)
    reward, done, info = assess(local_goal=(1, 0), end_time=25)
    assert (reward, done, type(info)) == (0, True, Timeout)
    assert not assess(end_time=24.75)[1]


def test_danger_uses_config_and_whole_sweep():
    reward, done, info = assess(humans=[((0.5, 0.7), (0.5, 0.7), 0.3)])
    assert not done and isinstance(info, Danger)
    assert info.min_sep_dynamic == pytest.approx(0.1)
    assert reward == pytest.approx(-0.1 * 0.7 * 0.25)


def test_old_info_constructors():
    for cls in (Collision, ReachGoal, Timeout, Nothing):
        assert cls().collision_kind is None
    assert Danger(0.1).min_dist == 0.1


def test_culled_static_clearance_matches_checking_every_rectangle():
    import numpy as np
    from crowd_sim.envs.utils.collision import rects_gap
    rng = np.random.RandomState(0)
    # land cells scattered over a 50 m map, the shape the maps produce
    cells = rng.randint(0, 50, size=(400, 2)).astype(float)
    rects = np.hstack([cells, cells + 1.0])
    for _ in range(2000):
        start = rng.uniform(0, 50, 2)
        end = start + rng.uniform(-3, 3, 2) * rng.randint(0, 2)
        radius = rng.uniform(0.1, 0.5)
        assert static_clearance(start, end, radius, rects) == pytest.approx(
            rects_gap(start, end, radius, rects), abs=1e-12)
    # a lone rectangle far beyond every search radius is still found
    far = np.array([[45.0, 45.0, 46.0, 46.0]])
    assert static_clearance((1, 1), (1, 1), 0.3, far) == pytest.approx(
        math.hypot(44, 44) - 0.3)
