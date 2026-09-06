"""Shared observation, prediction and execution for both simulation environments."""

import copy
import math

from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.collision import evaluate_motion
from crowd_sim.envs.utils.state import FullState, JointState, ObservableState


def advance(state, action, dt):
    if hasattr(action, 'vx'):
        vx, vy = action.vx, action.vy
        theta = getattr(state, 'theta', 0.0)
    else:
        theta = state.theta + action.r
        vx, vy = action.v * math.cos(theta), action.v * math.sin(theta)
    px, py = state.px + vx * dt, state.py + vy * dt
    if isinstance(state, FullState):
        return FullState(px, py, vx, vy, state.radius, state.gx, state.gy, state.v_pref,
                         theta % (2 * math.pi))
    return ObservableState(px, py, vx, vy, state.radius, is_static=state.is_static)


class TransitionMixin:
    def observe_at(self, position, dynamic=None):
        if self.robot.sensor != 'coordinates':
            raise NotImplementedError
        obs = ([h.get_observable_state() for h in self.humans]
               if dynamic is None else list(dynamic))
        obs.extend(h.get_observable_state() for h in getattr(self, 'humans_static', [])
                   if math.dist(h.get_position(), position) < self.nearby_dist)
        return obs

    def assess_motion(self, nav, next_nav, humans, next_humans, end_time=None):
        paths = [(h.position, n.position, h.radius) for h, n in zip(humans, next_humans)]
        return evaluate_motion(
            nav.position, next_nav.position, nav.radius, paths,
            getattr(self, 'land_rects', ()), getattr(self, 'map_bounds', None),
            nav.goal_position, getattr(self, 'robot_end', nav.goal_position),
            self.global_time + self.time_step if end_time is None else end_time,
            self.time_limit, self.time_step, self.success_reward, self.collision_penalty,
            self.discomfort_dist, self.discomfort_penalty_factor)

    def predict_humans(self, state, query=False):
        """Current and next observable states of the moving agents.

        With ``query`` the agents' own policies are previewed without changing
        the episode, exactly as ``onestep_lookahead`` does; otherwise observed
        velocities are extrapolated. The result does not depend on the robot's
        action, so a policy can compute it once and assess every candidate
        action against it with ``assess_motion``.
        """
        if query:
            actions, _, _ = self._human_moves(update=False)
            humans = [h.get_observable_state() for h in self.humans]
            return humans, [advance(h, a, self.time_step) for h, a in zip(humans, actions)]
        humans = [h for h in state.human_states if not h.is_static]
        return humans, [advance(h, ActionXY(h.vx, h.vy), self.time_step) for h in humans]

    def predict_linear(self, state, action):
        """Use observed velocities for prediction, plus known map geometry."""
        nav = state.self_state
        next_nav = advance(nav, action, self.time_step)
        humans, next_humans = self.predict_humans(state)
        reward, done, info = self.assess_motion(nav, next_nav, humans, next_humans)
        return self.observe_at(next_nav.position, next_humans), reward, done, info

    def _human_moves(self, update=False):
        """Use temporary policy state for previews and commit policies on execution."""
        goals = self.next_human_goals() if hasattr(self, 'next_human_goals') else None
        actions, states = [], []
        for i, human in enumerate(self.humans):
            nav = human.get_full_state()
            if goals is not None:
                nav.gx, nav.gy = goals[i][0]
                nav.goal_position = (nav.gx, nav.gy)
            obs = [h.get_observable_state() for h in self.humans if h is not human]
            if self.robot.visible:
                obs.append(self.robot.get_observable_state())
            state = JointState(nav, obs)
            if update:
                action = human.policy.predict(state)
            elif hasattr(human.policy, 'preview'):
                action = human.policy.preview(state)
            else:
                action = copy.deepcopy(human.policy).predict(state)
            if hasattr(self, 'keep_action_on_water'):
                action = self.keep_action_on_water(human, action)
            actions.append(action)
            states.append(state)
        return actions, states, goals

    def onestep_lookahead(self, action):
        return self.step(action, update=False)

    def step(self, action, update=True):
        actions, policy_states, goals = self._human_moves(update=update)
        nav = self.robot.get_full_state()
        next_nav = advance(nav, action, self.time_step)
        humans = [h.get_observable_state() for h in self.humans]
        next_humans = [advance(h, a, self.time_step) for h, a in zip(humans, actions)]
        reward, done, info = self.assess_motion(nav, next_nav, humans, next_humans)
        if update:
            self.states.append([nav, [h.get_full_state() for h in self.humans]])
            if hasattr(self.robot.policy, 'action_values'):
                self.action_values.append(self.robot.policy.action_values)
            if hasattr(self.robot.policy, 'get_attention_weights'):
                self.attention_weights.append(self.robot.policy.get_attention_weights())
            if goals is not None:
                self.commit_human_goals(goals)
            self.robot.step(action)
            for human, move, state in zip(self.humans, actions, policy_states):
                human.policy.last_state = state
                human.step(move)
            self.global_time += self.time_step
            for i, human in enumerate(self.humans):
                goal = (self.human_final_goal(i) if hasattr(self, 'human_final_goal')
                        else human.get_goal_position())
                if self.human_times[i] == 0 and math.dist(human.get_position(), goal) < human.radius:
                    self.human_times[i] = self.global_time
        return self.observe_at(next_nav.position, next_humans), reward, done, info
