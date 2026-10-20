import math

import numpy as np
import pytest

from crowd_nav.utils.local_goal import LocalGoalTracker
from crowd_nav.utils.theta_star import interpolate_path
from crowd_sim.envs.utils.collision import static_clearance
from crowd_sim.envs.utils.state import FullState, ObservableState


def ship(x, y, radius=0.3):
    return FullState(x, y, 0, 0, radius, 0, 0, 1, 0)


def mover(x, y, vx=1.0, vy=0.0):
    return ObservableState(x, y, vx, vy, 0.3)


def dense(points):
    return interpolate_path(points, step_size=0.2)


def test_progress_never_jumps_to_a_later_leg():
    # Looping route whose end passes 0.5 m from its first leg. A per-frame
    # nearest-point search from (5, 0.3) picks the end of the route and the
    # goal jumps there; the tracker keeps following the first leg.
    path = dense([(0, 0), (10, 0), (10, 3), (5, 3), (5, 0.5)])
    tracker = LocalGoalTracker(path, 0.25, goal_dist=1.0)
    goals, progress = [], []
    for x in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
        goals.append(tracker.update(ship(x, 0), []))
        progress.append(tracker.progress)
    assert progress == sorted(progress)
    assert all(g[1] == 0 for g in goals), 'goal must stay on the first leg'
    assert all(b[0] > a[0] for a, b in zip(goals, goals[1:]))
    before = tracker.progress
    goal = tracker.update(ship(5, 0.3), [])
    assert tracker.progress >= before
    assert goal[1] == 0 and goal[0] > 5
    assert math.dist((5, 0.3), path[-1]) < math.dist((5, 0.3), goal)


def test_hysteresis_and_rate_limited_lookahead():
    path = dense([(0, 0), (40, 0)])
    tracker = LocalGoalTracker(path, 0.25, goal_dist=1.0, monitor_dist=5.0,
                               monitor_exit=6.5, lookahead_rate=2.0)
    tracker.update(ship(0, 0), [mover(5.5, 0)])
    assert not tracker.monitoring and tracker.lookahead == 1.0
    tracker.update(ship(0, 0), [mover(4.9, 0)])
    assert tracker.monitoring
    # grows by lookahead_rate * dt per step instead of switching to 5 m at once
    assert tracker.lookahead == pytest.approx(1.5)
    assert math.dist((0, 0), tracker.goal) == pytest.approx(1.5, abs=0.2)
    for _ in range(20):
        tracker.update(ship(0, 0), [mover(4.9, 0)])
    assert tracker.lookahead == pytest.approx(5.0)
    # leaving: still monitored between the two radii, released beyond the exit radius
    tracker.update(ship(0, 0), [mover(6.0, 0)])
    assert tracker.monitoring and tracker.lookahead == pytest.approx(5.0)
    tracker.update(ship(0, 0), [mover(6.6, 0)])
    assert not tracker.monitoring and tracker.lookahead == pytest.approx(4.5)
    # static features and stopped ships are not monitored
    still = ObservableState(1, 0, 0, 0, 0.3, is_static=True)
    tracker.update(ship(0, 0), [still, mover(1, 0, 0, 0)])
    assert not tracker.monitoring


def test_goal_walks_back_before_a_blocked_corner():
    # Route bends around a land block; the straight line from the ship to a
    # point past the corner would cut across the land.
    land = [(3, -5, 8, -0.5)]
    path = dense([(0, -2), (2, -2), (2, 1), (10, 1)])
    tracker = LocalGoalTracker(path, 0.25, goal_dist=5.0, shore_clearance=0.5,
                               land_rects=land, map_bounds=(-10, -10, 20, 10))
    goal = tracker.update(ship(0, -2), [])
    corridor = 0.3 + 0.5
    assert static_clearance((0, -2), goal, corridor, land) >= 0
    assert tracker.min_goal_dist <= math.dist((0, -2), goal) < 5.0 - 0.2
    # the same route on open water reaches the full look-ahead, across the corner
    free = LocalGoalTracker(path, 0.25, goal_dist=5.0, shore_clearance=0.5)
    far = free.update(ship(0, -2), [])
    assert math.dist((0, -2), far) >= 5.0 - 0.2
    assert static_clearance((0, -2), far, corridor, land) < 0


def test_destination_only_needs_to_fit_the_ship():
    # land corner 0.72 m from the route end: inside the 0.8 m corridor, but the
    # ship (radius 0.3) still fits, so the destination is handed out as the goal
    land = [(10.6, -3, 12, -0.4)]
    path = dense([(0, 0), (10, 0)])
    tracker = LocalGoalTracker(path, 0.25, goal_dist=1.0, shore_clearance=0.5, land_rects=land)
    assert tracker.update(ship(7, 0), [])[0] == pytest.approx(8.0, abs=0.2)
    assert tracker.update(ship(9.2, 0), []) == (10, 0)
    assert static_clearance((9.2, 0), (10, 0), 0.3 + 0.5, land) < 0
    assert static_clearance((9.2, 0), (10, 0), 0.3, land) >= 0


def test_previous_goal_kept_when_nothing_is_visible():
    path = dense([(0, 0), (10, 0)])
    tracker = LocalGoalTracker(path, 0.25, goal_dist=2.0, land_rects=[(4, -1, 4.5, 1)])
    first = tracker.update(ship(0, 0), [])
    assert first[0] == pytest.approx(2.0, abs=0.2)
    # ship trapped behind the wall: every candidate segment crosses the land
    tracker.land_rects = np.array([[1, -1, 10, 1]], dtype=float)
    assert tracker.update(ship(0, 0), []) == first
    assert tracker.progress == 0


def test_final_goal_and_short_routes():
    path = dense([(0, 0), (3, 0)])
    tracker = LocalGoalTracker(path, 0.25, goal_dist=5.0)
    assert tracker.update(ship(0, 0), []) == (3, 0)
    assert tracker.update(ship(2.9, 0), []) == (3, 0)
    with pytest.raises(ValueError):
        LocalGoalTracker([], 0.25)
    with pytest.raises(ValueError):
        LocalGoalTracker(path, 0.25, monitor_dist=5.0, monitor_exit=4.0)


def test_from_env_uses_map_geometry(make_env):
    env = make_env(obstacles=[dict(cx=7.5, cy=5.5, w=1, h=1)])
    path = dense(env.global_path)[1:]
    tracker = LocalGoalTracker.from_env(env, path, goal_dist=4.0)
    assert tracker.time_step == env.time_step
    assert tracker.land_rects.shape == (1, 4) and tracker.map_bounds == (0, 0, 50, 50)
    goal = tracker.update(env.robot.get_full_state(), [])
    # the 1 m block at x in [7, 8] sits on the straight route, so the goal stops short of it
    assert goal[0] < 7 - env.robot.radius
