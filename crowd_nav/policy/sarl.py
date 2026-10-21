import logging
import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn

from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.collision import circle_clearance, static_clearance
from crowd_sim.envs.utils.info import Collision
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.transition import advance


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims, attention_dims, with_global_state,
                 cell_size, cell_num):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        self.cell_size = cell_size
        self.cell_num = cell_num
        mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)
        self.attention_weights = None

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        valid = state[:, :, self.self_state_dim:].abs().sum(dim=2) > 0
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)

        if self.with_global_state:
            # compute attention scores
            embeddings = mlp1_output.view(size[0], size[1], -1)
            count = valid.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(2)
            global_state = (embeddings * valid.unsqueeze(2)).sum(dim=1, keepdim=True) / count
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
                contiguous().view(-1, self.global_state_dim)
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)

        # Mask explicit empty interaction rows, never numerical score values.
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1) * valid
        weights = weights.unsqueeze(2)
        self.attention_weights = weights[0, valid[0], 0].detach().cpu().numpy()

        # output feature is a linear combination of input features
        features = mlp2_output.view(size[0], size[1], -1)
        # for converting to onnx
        # expanded_weights = torch.cat([torch.zeros(weights.size()).copy_(weights) for _ in range(50)], dim=2)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        value = self.mlp3(joint_state)
        return value


# One candidate action after the swept safety check. ``clearance`` is the worst
# margin-adjusted gap over the next step and the braking manoeuvre after it;
# negative means the candidate is unsafe. ``speed`` and ``velocity_change`` are
# the speed after the action and the size of the velocity jump it causes;
# ``heading_change`` is the turn of the direction of motion in radians, zero
# for a ship setting off from rest and for the stop action.
Candidate = namedtuple('Candidate', ['index', 'action', 'next_nav', 'reward', 'done', 'info',
                                     'clearance', 'safe', 'speed', 'velocity_change',
                                     'heading_change'])


class SARL(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'SARL'
        # kinematic limits used to build the candidate actions, see configure()
        self.constrained = True
        self.max_accel = 0.5
        self.max_decel = 0.5
        self.max_turn_rate = math.radians(60)
        self.heading_samples = 7
        self.stop_speed = 0.05
        # value penalty per radian of heading change, see predict()
        self.continuity_weight = 0.01
        # safety filter
        self.safety_margin_static = 0.2
        self.safety_margin_dynamic = 0.2
        self.braking_check = True
        self.safety_horizon = 3.0
        # land margin in force for the step being predicted, see coast_margin()
        self.step_static_margin = None
        # diagnostics of the last prediction
        self.candidates = None
        self.safe_mask = None
        self.emergency = False

    def configure(self, config):
        self.set_common_parameters(config)
        mlp1_dims = [int(x) for x in config.get('sarl', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('sarl', 'mlp2_dims').split(', ')]
        mlp3_dims = [int(x) for x in config.get('sarl', 'mlp3_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('sarl', 'attention_dims').split(', ')]
        self.with_om = config.getboolean('sarl', 'with_om')
        with_global_state = config.getboolean('sarl', 'with_global_state')
        self.model = ValueNetwork(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
                                  attention_dims, with_global_state, self.cell_size, self.cell_num)
        self.multiagent_training = config.getboolean('sarl', 'multiagent_training')
        if self.with_om:
            self.name = 'OM-SARL'
        self.configure_motion(config)
        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))

    def configure_motion(self, config):
        """Read the [motion] and [safety] sections; older configs keep the defaults."""
        self.constrained = config.getboolean('motion', 'constrained', fallback=self.constrained)
        self.max_accel = config.getfloat('motion', 'max_accel', fallback=self.max_accel)
        self.max_decel = config.getfloat('motion', 'max_decel', fallback=self.max_decel)
        self.max_turn_rate = math.radians(config.getfloat(
            'motion', 'max_turn_rate', fallback=math.degrees(self.max_turn_rate)))
        self.heading_samples = config.getint('motion', 'heading_samples', fallback=self.heading_samples)
        self.stop_speed = config.getfloat('motion', 'stop_speed', fallback=self.stop_speed)
        self.continuity_weight = config.getfloat('motion', 'continuity_weight',
                                                 fallback=self.continuity_weight)
        self.safety_margin_static = config.getfloat('safety', 'safety_margin_static',
                                                    fallback=self.safety_margin_static)
        self.safety_margin_dynamic = config.getfloat('safety', 'safety_margin_dynamic',
                                                     fallback=self.safety_margin_dynamic)
        self.braking_check = config.getboolean('safety', 'braking_check', fallback=self.braking_check)
        self.safety_horizon = config.getfloat('safety', 'safety_horizon', fallback=self.safety_horizon)
        if min(self.max_accel, self.max_decel, self.max_turn_rate) <= 0:
            raise ValueError('max_accel, max_decel and max_turn_rate must be positive')
        if self.safety_horizon < 0:
            raise ValueError('safety_horizon must not be negative')
        if self.heading_samples < 1 or self.speed_samples < 1:
            raise ValueError('heading_samples and speed_samples must be positive')
        logging.info('Motion limits: %s, accel %.2f, decel %.2f, turn %.0f deg/s; safety margins '
                     'static %.2f dynamic %.2f, braking check %s over %.1f s',
                     'on' if self.constrained else 'off', self.max_accel, self.max_decel,
                     math.degrees(self.max_turn_rate), self.safety_margin_static,
                     self.safety_margin_dynamic, 'on' if self.braking_check else 'off',
                     self.safety_horizon)

    def get_attention_weights(self):
        return self.model.attention_weights

    # -- candidate actions ----------------------------------------------------

    def stop_action(self):
        return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)

    def candidate_actions(self, nav):
        """Actions the ship can execute in the next step from its current velocity.

        Speeds stay within the acceleration and deceleration limits and the
        direction of motion within the turn rate limit, so every candidate can
        be executed as chosen. The current speed and heading are always among
        the samples, and the stop action is added whenever it is reachable. A
        ship that is (nearly) at rest may set off in any direction. A preset
        ``action_space`` is used unchanged, which is also the fallback when the
        limits are switched off.
        """
        if self.action_space is not None:
            return list(self.action_space)
        if not self.constrained:
            self.build_action_space(nav.v_pref)
            return list(self.action_space)

        dt = self.time_step
        speed = math.hypot(nav.vx, nav.vy)
        low = max(0.0, speed - self.max_decel * dt)
        high = min(nav.v_pref, speed + self.max_accel * dt)
        speeds = set(np.linspace(low, high, self.speed_samples).tolist())
        if low < speed < high:
            speeds.add(speed)
        speeds = sorted(s for s in speeds if s > 1e-9)
        turn = self.max_turn_rate * dt
        offsets = sorted(set(np.linspace(-turn, turn, self.heading_samples).tolist()) | {0.0})

        actions = []
        if self.kinematics == 'holonomic':
            if speed < self.stop_speed:
                headings = np.linspace(0, 2 * np.pi, self.rotation_samples, endpoint=False)
            else:
                headings = math.atan2(nav.vy, nav.vx) + np.array(offsets)
            for heading in headings:
                for s in speeds:
                    actions.append(ActionXY(s * math.cos(heading), s * math.sin(heading)))
        else:
            rotations = (np.linspace(-np.pi / 4, np.pi / 4, self.rotation_samples)
                         if speed < self.stop_speed else offsets)
            for rotation in rotations:
                for s in speeds:
                    actions.append(ActionRot(s, rotation))
        if low <= 1e-9:
            actions.append(self.stop_action())
        return actions

    # -- safety filter --------------------------------------------------------

    COAST_SLACK = 0.01  # m a ship inside the land margin may close its gap by per step

    def coast_margin(self, nav):
        """Land margin for a step that starts at ``nav``.

        Every candidate is swept from the current position, so a ship that is
        already closer to land than ``safety_margin_static`` fails the check
        whatever it does and stays in the emergency fallback, where the value
        network is never consulted. Inside the margin the ship's own gap is
        used instead: holding or opening it passes, closing it does not.
        """
        gap = static_clearance(nav.position, nav.position, nav.radius,
                               getattr(self.env, 'land_rects', ()), getattr(self.env, 'map_bounds', None))
        return max(0.0, min(self.safety_margin_static, gap - self.COAST_SLACK))

    def land_margin(self):
        """The land margin in force: the step's own while predicting, else the configured one."""
        return self.safety_margin_static if self.step_static_margin is None else self.step_static_margin

    def braking_clearance(self, nav, humans):
        """Worst margin-adjusted gap while stopping from ``nav`` at ``max_decel``.

        The ship keeps its heading and the other agents their velocity, which is
        the same constant velocity model the one-step sweep uses. Every braking
        substep is swept with the shared geometry, so an action that is clear
        for one step but leaves no room to stop afterwards is rejected too.

        Being stopped is not assumed safe: once at rest the ship must stay clear
        of the extrapolated ships until ``safety_horizon`` seconds have passed,
        so slowing to a halt in the path of an oncoming ship is rejected while
        there is still speed to dodge with. Land cannot reach a stopped ship,
        so the static check ends with the braking.
        """
        dt = self.time_step
        speed = math.hypot(nav.vx, nav.vy)
        heading = np.array((nav.vx, nav.vy)) / speed if speed > 0 else np.zeros(2)
        rects = getattr(self.env, 'land_rects', ())
        bounds = getattr(self.env, 'map_bounds', None)
        position = np.array(nav.position, dtype=float)
        others = [(np.array(h.position, dtype=float), np.array(h.velocity, dtype=float) * dt, h.radius)
                  for h in humans]
        clearance = math.inf
        elapsed = 0.0
        braking_start = position
        while speed > 0:
            next_speed = max(0.0, speed - self.max_decel * dt)
            end = position + heading * (speed + next_speed) / 2 * dt
            for k, (start, move, radius) in enumerate(others):
                clearance = min(clearance, circle_clearance(position, end, nav.radius, start,
                                                            start + move, radius)
                                - self.safety_margin_dynamic)
                others[k] = (start + move, move, radius)
            position, speed, elapsed = end, next_speed, elapsed + dt
        if elapsed > 0:
            # The heading is held, so the braking path is one straight segment and a
            # single sweep finds the same land gap as sweeping every substep.
            clearance = min(clearance, static_clearance(braking_start, position, nav.radius, rects,
                                                        bounds) - self.land_margin())
        remaining = self.safety_horizon - elapsed
        if remaining > 0:
            for start, move, radius in others:
                clearance = min(clearance, circle_clearance(position, position, nav.radius, start,
                                                            start + move * remaining / dt, radius)
                                - self.safety_margin_dynamic)
        return clearance

    def check_action(self, index, nav, action, humans, next_humans):
        """Sweep one candidate over the next step (and the braking after it)."""
        next_nav = advance(nav, action, self.time_step)
        reward, done, info = self.env.assess_motion(nav, next_nav, humans, next_humans)
        clearance = min(info.min_sep_static - self.land_margin(),
                        info.min_sep_dynamic - self.safety_margin_dynamic)
        if self.braking_check and not done:
            clearance = min(clearance, self.braking_clearance(next_nav, next_humans))
        safe = clearance >= 0 and not isinstance(info, Collision)
        speed = math.hypot(next_nav.vx, next_nav.vy)
        change = math.hypot(next_nav.vx - nav.vx, next_nav.vy - nav.vy)
        turn = 0.0
        if math.hypot(nav.vx, nav.vy) >= self.stop_speed and speed > 1e-9:
            turn = abs(math.remainder(math.atan2(next_nav.vy, next_nav.vx)
                                      - math.atan2(nav.vy, nav.vx), 2 * math.pi))
        return Candidate(index, action, next_nav, reward, done, info, clearance, safe, speed, change,
                         turn)

    def held_clearance(self, nav, candidate, others):
        """Worst margin-adjusted gap if the candidate's velocity is held, not braked.

        The braking check assumes the ship stops after the step, which is the
        wrong model when another ship does not give way: braking in its path
        only lets it arrive. Here the candidate velocity is held for
        ``safety_horizon`` seconds against the extrapolated ships. Land is
        checked one second ahead only, since the ship can still turn from it.
        """
        horizon = self.safety_horizon
        start = candidate.next_nav.position
        vx, vy = candidate.next_nav.vx, candidate.next_nav.vy
        held = (start[0] + vx * horizon, start[1] + vy * horizon)
        clearance = static_clearance(start, (start[0] + vx, start[1] + vy), nav.radius,
                                     getattr(self.env, 'land_rects', ()),
                                     getattr(self.env, 'map_bounds', None)) - self.land_margin()
        for other in others:
            end = (other.px + other.vx * horizon, other.py + other.vy * horizon)
            clearance = min(clearance, circle_clearance(start, held, nav.radius, other.position, end,
                                                        other.radius) - self.safety_margin_dynamic)
        return clearance

    def emergency_action(self, nav, candidates, others=()):
        """Best evasive action when no candidate is safe.

        An action that collides within the next step is never chosen while
        another one does not. Among the rest the action that stays clearest
        with its velocity held (see ``held_clearance``) wins, accelerating
        ones included, so the ship dodges an oncoming ship instead of braking
        in its path; ties go to the smallest change of velocity.
        """
        def rank(c):
            return (not isinstance(c.info, Collision), self.held_clearance(nav, c, others),
                    -c.velocity_change)
        return max(candidates, key=rank).action

    # -- prediction -----------------------------------------------------------

    def predict(self, state):
        """
        Score only the candidates that pass the swept safety check with the value
        network, charge ``continuity_weight`` per radian of heading change so
        close scores keep the current course, and fall back to the best evasive
        action when nothing is safe. The chosen action is executed exactly
        as scored; nothing is smoothed afterwards.

        Reaching the goal in ``state`` (usually a local goal on the route) does
        not short-circuit to a stop: stopping is a candidate like any other and
        has to pass the same safety check, so the ship does not park in front
        of an oncoming ship. The environment ends the episode at the final goal.
        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')
        self.emergency = False
        self.step_static_margin = self.coast_margin(state.self_state)
        try:
            return self.select_action(state)
        finally:
            self.step_static_margin = None

    def select_action(self, state):
        """The body of ``predict``, run with the step's land margin in force."""
        nav = state.self_state
        humans, next_humans = self.env.predict_humans(state, self.query_env)
        actions = self.candidate_actions(nav)
        candidates = [self.check_action(i, nav, a, humans, next_humans) for i, a in enumerate(actions)]
        safe = [c for c in candidates if c.safe]
        self.candidates = actions
        self.safe_mask = [c.safe for c in candidates]
        # Excluded candidates are recorded as if they collided.
        self.action_values = [self.env.collision_penalty] * len(actions)
        self.emergency = not safe

        if not safe:
            action = self.emergency_action(nav, candidates, next_humans)
        elif self.phase == 'train' and np.random.random() < self.epsilon:
            action = safe[np.random.choice(len(safe))].action
        else:
            gamma_bar = pow(self.gamma, self.time_step * nav.v_pref)
            best_score, action = -math.inf, None
            for candidate in safe:
                value = candidate.reward
                if not candidate.done:
                    observation = self.env.observe_at(candidate.next_nav.position, next_humans)
                    next_input = self.transform(JointState(candidate.next_nav, observation)).unsqueeze(0)
                    with torch.no_grad():
                        value += gamma_bar * self.model(next_input).item()
                self.action_values[candidate.index] = value
                score = value - self.continuity_weight * candidate.heading_change
                if score > best_score:
                    best_score, action = score, candidate.action

        if self.phase == 'train':
            self.last_state = self.transform(state)
        return action
