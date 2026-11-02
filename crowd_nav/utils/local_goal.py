"""Local goal selection along a planned global route.

The tracker turns a dense, land free route into a sequence of local goals that
move forward smoothly. It replaces the stateless per-frame search that picked
the globally nearest route point: on a winding route that point can lie on a
later leg, so the goal jumped backwards or across a headland.
"""
import math

import numpy as np

from crowd_sim.envs.utils.collision import static_clearance


class LocalGoalTracker:
    """Pick a local goal on ``path`` that is continuous and reachable.

    * Progress along the route is remembered, and the route is only searched
      forward of it within ``window`` metres of arc length. Progress never moves
      backwards, and it only advances to a route point the ship can see.
    * The look-ahead distance is ``goal_dist`` on open water and ``monitor_dist``
      while moving obstacles are monitored. Monitoring starts when an obstacle
      comes within ``monitor_dist`` and stops only once every obstacle is beyond
      ``monitor_exit``. The look-ahead moves towards its target by at most
      ``lookahead_rate`` metres per second instead of switching.
    * A candidate goal is used only when the straight segment from the ship to
      it keeps ``shore_clearance`` from land and from the map edge. When the
      corridor is blocked the goal walks back towards the ship; when nothing
      is visible the previous goal is kept as long as it is still ahead. The
      destination itself only has to fit the ship, so a route that ends near
      the coast can still be completed.
    * A ship that has stood still for ``stall_time`` seconds on the goal it kept
      would stay there, since nothing ahead passes the corridor check and
      stopping on a goal scores well. Only then the corridor narrows to the land
      margin of the safety filter (see ``SARL.coast_margin``), and as a last
      resort to the ship itself. ``stall_time = 0`` turns this off.
    """

    COAST_SLACK = 0.01  # m the narrowed corridor may close the ship's own land gap by

    def __init__(self, path, time_step, goal_dist=1.0, monitor_dist=5.0, monitor_exit=None,
                 lookahead_rate=1.0, shore_clearance=0.5, min_goal_dist=0.5, land_rects=(),
                 map_bounds=None, window=None, visibility_checks=10, stall_time=10.0,
                 stop_speed=0.05, safety_margin_static=0.2):
        self.path = [tuple(map(float, p)) for p in path]
        if not self.path:
            raise ValueError('The route needs at least one point')
        if monitor_exit is None:
            monitor_exit = 1.3 * monitor_dist
        if monitor_exit < monitor_dist:
            raise ValueError('monitor_exit must not be smaller than monitor_dist')
        if lookahead_rate <= 0 or time_step <= 0:
            raise ValueError('lookahead_rate and time_step must be positive')
        self.time_step = time_step
        self.goal_dist = goal_dist
        self.monitor_dist = monitor_dist
        self.monitor_exit = monitor_exit
        self.lookahead_rate = lookahead_rate
        self.shore_clearance = shore_clearance
        self.min_goal_dist = min_goal_dist
        self.land_rects = np.asarray(land_rects, dtype=float).reshape(-1, 4)
        self.map_bounds = map_bounds
        self.window = 2 * monitor_dist if window is None else window
        self.visibility_checks = visibility_checks
        self.stall_time = stall_time
        self.stop_speed = stop_speed
        self.safety_margin_static = safety_margin_static
        steps = [math.dist(a, b) for a, b in zip(self.path, self.path[1:])]
        self.arc = np.concatenate([[0.0], np.cumsum(steps)])
        # state carried from step to step
        self.progress = 0
        self.lookahead = goal_dist
        self.monitoring = False
        self.goal_index = None
        self.goal = self.path[0]
        self.nearest_obstacle = math.inf
        self.stopped_time = 0.0

    @classmethod
    def from_env(cls, env, path, **kwargs):
        """Build a tracker that uses the land geometry of a map environment.

        The stall settings come from the policy sailing the ship, when it has them.
        """
        policy = getattr(getattr(env, 'robot', None), 'policy', None)
        for name in ('stall_time', 'stop_speed', 'safety_margin_static'):
            if hasattr(policy, name):
                kwargs.setdefault(name, getattr(policy, name))
        return cls(path, env.time_step, land_rects=getattr(env, 'land_rects', ()),
                   map_bounds=getattr(env, 'map_bounds', None), **kwargs)

    def update(self, ship_state, obstacle_states):
        """Advance the tracker one step and return the local goal as (x, y)."""
        position = (ship_state.px, ship_state.py)
        stopped = math.hypot(ship_state.vx, ship_state.vy) < self.stop_speed
        self.stopped_time = self.stopped_time + self.time_step if stopped else 0.0
        self._advance_progress(position, ship_state.radius)
        self._update_lookahead(position, obstacle_states)
        self.goal = self._select_goal(position, ship_state.radius)
        return self.goal

    # -- pieces of update() -------------------------------------------------

    def _clear(self, start, end, radius):
        return static_clearance(start, end, radius, self.land_rects, self.map_bounds) >= 0

    def _window_end(self):
        end = self.progress
        limit = self.arc[self.progress] + self.window
        while end + 1 < len(self.path) and self.arc[end + 1] <= limit:
            end += 1
        return end

    def _advance_progress(self, position, radius):
        """Move progress to the nearest visible route point ahead of it."""
        candidates = range(self.progress, self._window_end() + 1)
        order = sorted(candidates, key=lambda i: math.dist(position, self.path[i]))
        for i in order[:self.visibility_checks]:
            if self._clear(position, self.path[i], radius):
                self.progress = i
                return

    @staticmethod
    def moving(obstacle_states):
        for state in obstacle_states:
            if getattr(state, 'is_static', False) or abs(state.vx) + abs(state.vy) < 1e-10:
                continue
            yield state

    def _update_lookahead(self, position, obstacle_states):
        self.nearest_obstacle = min((math.dist((o.px, o.py), position)
                                     for o in self.moving(obstacle_states)), default=math.inf)
        threshold = self.monitor_exit if self.monitoring else self.monitor_dist
        self.monitoring = self.nearest_obstacle < threshold
        target = self.monitor_dist if self.monitoring else self.goal_dist
        step = self.lookahead_rate * self.time_step
        self.lookahead += float(np.clip(target - self.lookahead, -step, step))

    def _select_goal(self, position, radius):
        last = len(self.path) - 1
        ahead = last
        for i in range(self.progress, last + 1):
            if math.dist(position, self.path[i]) >= self.lookahead:
                ahead = i
                break
        for clearance in self._clearances(position, radius):
            for i in range(ahead, self.progress - 1, -1):
                if i != last and math.dist(position, self.path[i]) < self.min_goal_dist:
                    break
                if self._clear(position, self.path[i], radius if i == last else radius + clearance):
                    self.goal_index = i
                    return self.path[i]
        if self.goal_index is not None and self.goal_index >= self.progress:
            return self.path[self.goal_index]
        self.goal_index = ahead
        return self.path[ahead]

    def _parked(self, position):
        """True when the ship has stood still for ``stall_time`` on the goal it kept."""
        return (0 < self.stall_time <= self.stopped_time
                and self.goal_index is not None and self.goal_index >= self.progress
                and math.dist(position, self.path[self.goal_index]) < self.min_goal_dist)

    def _clearances(self, position, radius):
        """Room to keep from land on the way to a goal, strictest first."""
        if not self._parked(position):
            return [self.shore_clearance]
        gap = static_clearance(position, position, radius, self.land_rects, self.map_bounds)
        margin = min(self.shore_clearance, self.safety_margin_static, gap - self.COAST_SLACK)
        return [self.shore_clearance, max(0.0, margin), 0.0]
