"""Analytic swept clearances and shared transition outcomes."""

import math

import numpy as np

from crowd_sim.envs.utils.info import Collision, Danger, Nothing, ReachGoal, Timeout


def circle_clearance(start, end, radius, other_start, other_end, other_radius):
    """Minimum surface gap between two circles moving linearly over one step."""
    p = np.asarray(other_start, dtype=float) - start
    d = np.asarray(other_end, dtype=float) - end - p
    den = float(d @ d)
    t = np.clip(-float(p @ d) / den, 0.0, 1.0) if den else 0.0
    return float(np.linalg.norm(p + t * d) - radius - other_radius)


def static_clearance(start, end, radius, rects=(), bounds=None):
    """Distance from a swept circle to closed rectangles and the map exterior.

    Rectangles and bounds use (xmin, ymin, xmax, ymax). A path intersecting
    a filled rectangle has gap -radius; this is not a penetration-depth metric.
    """
    p, q = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    gap = math.inf
    boxes = np.asarray(rects, dtype=float).reshape(-1, 4)
    if len(boxes):
        lo, hi = boxes[:, :2], boxes[:, 2:]
        # Endpoint-to-box and corner-to-segment distances cover disjoint segments.
        dp = np.maximum(np.maximum(lo - p, p - hi), 0.0)
        dq = np.maximum(np.maximum(lo - q, q - hi), 0.0)
        dist2 = np.minimum(np.sum(dp * dp, axis=1), np.sum(dq * dq, axis=1))
        d = q - p
        den = float(d @ d)
        for x, y in ((0, 1), (0, 3), (2, 1), (2, 3)):
            corners = boxes[:, [x, y]]
            t = np.clip((corners - p) @ d / den, 0.0, 1.0) if den else np.zeros(len(boxes))
            delta = corners - (p + t[:, None] * d)
            dist2 = np.minimum(dist2, np.sum(delta * delta, axis=1))
        # Slab intersection detects crossing even when both endpoints are outside.
        enter, leave = np.zeros(len(boxes)), np.ones(len(boxes))
        intersects = np.ones(len(boxes), dtype=bool)
        for axis in range(2):
            if d[axis] == 0:
                intersects &= (p[axis] >= lo[:, axis]) & (p[axis] <= hi[:, axis])
            else:
                a = (lo[:, axis] - p[axis]) / d[axis]
                b = (hi[:, axis] - p[axis]) / d[axis]
                enter = np.maximum(enter, np.minimum(a, b))
                leave = np.minimum(leave, np.maximum(a, b))
        dist2[intersects & (enter <= leave)] = 0.0
        gap = float(np.sqrt(np.min(dist2)) - radius)
    if bounds is not None:
        xmin, ymin, xmax, ymax = bounds
        edge_gap = min(p[0] - xmin, q[0] - xmin, xmax - p[0], xmax - q[0],
                       p[1] - ymin, q[1] - ymin, ymax - p[1], ymax - q[1]) - radius
        gap = min(gap, float(edge_gap))
    return gap


def evaluate_motion(start, end, radius, humans, rects, bounds, local_goal, final_goal,
                    end_time, time_limit, time_step, success_reward, collision_penalty,
                    discomfort_dist, discomfort_penalty_factor):
    """Evaluate geometry, reward and termination without changing state.

    Each human is (start_position, end_position, radius). Local arrival shapes
    the reward; only final arrival terminates the episode.
    """
    dyn = min((circle_clearance(start, end, radius, a, b, r) for a, b, r in humans),
              default=math.inf)
    stat = static_clearance(start, end, radius, rects, bounds)
    kind = 'both' if dyn < 0 and stat < 0 else 'dynamic' if dyn < 0 else 'static' if stat < 0 else None
    meta = dict(min_sep_dynamic=dyn, min_sep_static=stat, collision_kind=kind)
    gap = min(dyn, stat)
    local = math.dist(end, local_goal) < radius
    final = math.dist(end, final_goal) < radius
    if kind:
        return collision_penalty, True, Collision(**meta)
    if final:
        return success_reward, True, ReachGoal(**meta)
    if end_time >= time_limit or math.isclose(end_time, time_limit, rel_tol=0, abs_tol=1e-9):
        return 0.0, True, Timeout(**meta)
    if gap < discomfort_dist:
        reward = (success_reward if local else
                  (gap - discomfort_dist) * discomfort_penalty_factor * time_step)
        return reward, False, Danger(gap, **meta)
    return success_reward if local else 0.0, False, Nothing(**meta)
