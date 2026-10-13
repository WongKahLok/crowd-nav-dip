import math

import numpy as np
import pytest

from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import Collision
from crowd_sim.envs.utils.state import JointState

from conftest import FixedPolicy


def moving(state, vx, vy):
    state.vx, state.vy = vx, vy
    state.velocity = (vx, vy)
    return state


def speeds_and_headings(actions):
    moving_actions = [a for a in actions if math.hypot(a.vx, a.vy) > 1e-9]
    return ([math.hypot(a.vx, a.vy) for a in moving_actions],
            [math.atan2(a.vy, a.vx) for a in moving_actions])


def test_candidates_respect_limits_and_keep_current_motion(make_env, make_sarl):
    env = make_env()
    policy = make_sarl(env)
    dt = env.time_step
    nav = moving(env.robot.get_full_state(), 0.6, 0.0)
    actions = policy.candidate_actions(nav)
    speeds, headings = speeds_and_headings(actions)
    assert min(speeds) >= 0.6 - policy.max_decel * dt - 1e-9
    assert max(speeds) <= 0.6 + policy.max_accel * dt + 1e-9
    assert max(abs(h) for h in headings) <= policy.max_turn_rate * dt + 1e-9
    assert ActionXY(0.6, 0.0) in actions, 'keeping the current velocity is always a candidate'
    assert ActionXY(0, 0) not in actions, 'a full stop is not reachable from 0.6 m/s in one step'
    assert len(set(round(h, 9) for h in headings)) == policy.heading_samples
    # a ship at rest may set off in any direction, but only as fast as it can accelerate
    rest = policy.candidate_actions(moving(env.robot.get_full_state(), 0, 0))
    speeds, headings = speeds_and_headings(rest)
    assert ActionXY(0, 0) in rest
    assert max(speeds) <= policy.max_accel * dt + 1e-9
    assert len(set(round(h, 9) for h in headings)) == policy.rotation_samples
    # the stop action appears as soon as it is reachable
    slow = policy.candidate_actions(moving(env.robot.get_full_state(), 0.1, 0))
    assert ActionXY(0, 0) in slow
    # the speed never exceeds v_pref
    fast = policy.candidate_actions(moving(env.robot.get_full_state(), 1.0, 0))
    assert max(speeds_and_headings(fast)[0]) <= nav.v_pref + 1e-9


def test_unconstrained_mode_and_preset_action_space(make_env, make_sarl):
    env = make_env()
    policy = make_sarl(env)
    policy.constrained = False
    nav = moving(env.robot.get_full_state(), 0.6, 0.0)
    assert len(policy.candidate_actions(nav)) == policy.speed_samples * policy.rotation_samples + 1
    policy.action_space = [ActionXY(1, 0)]
    assert policy.candidate_actions(nav) == [ActionXY(1, 0)]


def test_unicycle_candidates(make_env, make_sarl):
    env = make_env()
    policy = make_sarl(env)
    policy.kinematics = 'unicycle'
    dt = env.time_step
    nav = moving(env.robot.get_full_state(), 0.6, 0.0)
    actions = policy.candidate_actions(nav)
    assert all(isinstance(a, ActionRot) for a in actions)
    assert max(abs(a.r) for a in actions) <= policy.max_turn_rate * dt + 1e-9
    assert ActionRot(0.6, 0.0) in actions


def test_chosen_action_is_executed_unchanged(make_env, make_sarl):
    env = make_env(dynamic=[dict(px=9, py=5, gx=2, gy=5, path=[(9, 5), (2, 5)])])
    policy = make_sarl(env)
    for _ in range(6):
        ob = env.light_reset()
        action = env.robot.act(ob)
        assert action in policy.candidates
        assert policy.safe_mask[policy.candidates.index(action)] or policy.emergency
        env.step(action)
        assert env.robot.get_velocity() == (action.vx, action.vy)


def test_safety_filter_uses_margin_and_scores_only_safe_actions(make_env, make_sarl):
    # land from x = 5.85: at 1 m/s the sweep ends with a 0.3 m gap, which is
    # below the 0.5 m margin, so that action is unsafe without colliding
    env = make_env(obstacles=[dict(cx=6.35, cy=5, w=1, h=1)])
    policy = make_sarl(env)
    policy.safety_margin_static = 0.5
    policy.braking_check = False
    policy.action_space = [ActionXY(1, 0), ActionXY(0, 1), ActionXY(0, 0)]
    calls = []
    hook = policy.model.register_forward_hook(lambda *args: calls.append(True))
    action = env.robot.act(env.observe_at(env.robot.get_position()))
    hook.remove()
    assert policy.safe_mask == [False, True, True]
    assert len(calls) == 2, 'only safe candidates are scored'
    assert action in (ActionXY(0, 1), ActionXY(0, 0))
    assert policy.action_values[0] == env.collision_penalty
    assert not policy.emergency
    policy.safety_margin_static = 0.2
    env.robot.act(env.observe_at(env.robot.get_position()))
    assert policy.safe_mask == [True, True, True]


def test_braking_check_rejects_actions_that_cannot_stop(make_env, make_sarl):
    # land from x = 6.5. From 1 m/s the next step ends with a 0.95 m gap, but
    # stopping at 0.5 m/s^2 needs a further 1 m plus the 0.2 m margin.
    env = make_env(obstacles=[dict(cx=7.0, cy=5, w=1, h=1)])
    policy = make_sarl(env)
    policy.action_space = [ActionXY(1, 0), ActionXY(0.5, 0)]
    nav = moving(env.robot.get_full_state(), 1.0, 0.0)
    humans, next_humans = env.predict_humans(JointState(nav, []))
    fast = policy.check_action(0, nav, ActionXY(1, 0), humans, next_humans)
    assert fast.info.min_sep_static > policy.safety_margin_static
    assert not fast.safe and fast.clearance < 0
    slow = policy.check_action(1, nav, ActionXY(0.5, 0), humans, next_humans)
    assert slow.safe
    policy.braking_check = False
    assert policy.check_action(0, nav, ActionXY(1, 0), humans, next_humans).safe
    assert math.isinf(policy.braking_clearance(moving(env.robot.get_full_state(), 0, 0), []))


def test_stopping_in_the_path_of_a_ship_is_unsafe(make_env, make_sarl, env_cfg):
    env = make_env()
    policy = make_sarl(env)
    # a ship 2 m ahead heading straight at the resting robot at 1 m/s
    oncoming = Human(env_cfg, 'humans')
    oncoming.set(7.3, 5.0, 0, 5, -1, 0, 0)
    nav = env.robot.get_full_state()
    humans, next_humans = env.predict_humans(JointState(nav, [oncoming.get_observable_state()]))
    stop = policy.check_action(0, nav, ActionXY(0, 0), humans, next_humans)
    assert stop.info.min_sep_dynamic > policy.safety_margin_dynamic, 'the next step alone is fine'
    assert not stop.safe, 'waiting for the ship to arrive is not safe'
    policy.safety_horizon = 1.0
    assert policy.check_action(0, nav, ActionXY(0, 0), humans, next_humans).safe
    policy.safety_horizon = 3.0
    # a ship that passes 1 m to the side leaves the stopped robot alone
    passing = Human(env_cfg, 'humans')
    passing.set(7.3, 6.0, 0, 6, -1, 0, 0)
    humans, next_humans = env.predict_humans(JointState(nav, [passing.get_observable_state()]))
    assert policy.check_action(0, nav, ActionXY(0, 0), humans, next_humans).safe


def test_braking_check_sees_moving_ships(make_env, make_sarl, env_cfg):
    env = make_env()
    policy = make_sarl(env)
    nav = moving(env.robot.get_full_state(), 1.0, 0.0)
    # a ship crossing the braking path 1.2 m ahead in 1 s
    crossing = Human(env_cfg, 'humans')
    crossing.set(6.2, 6.5, 6.2, 0, 0, -1.5, 0)
    state = JointState(nav, [crossing.get_observable_state()])
    humans, next_humans = env.predict_humans(state)
    candidate = policy.check_action(0, nav, ActionXY(1, 0), humans, next_humans)
    assert candidate.info.min_sep_dynamic > policy.safety_margin_dynamic, 'next step alone is fine'
    assert not candidate.safe


def emergency_candidates(policy, env, nav, ships):
    humans, next_humans = env.predict_humans(JointState(nav, [s.get_observable_state() for s in ships]))
    candidates = [policy.check_action(i, nav, a, humans, next_humans)
                  for i, a in enumerate(policy.action_space)]
    return [c._replace(safe=False) for c in candidates], next_humans


def test_emergency_fallback_dodges_instead_of_braking(make_env, make_sarl, env_cfg):
    env = make_env()
    policy = make_sarl(env)
    policy.action_space = [ActionXY(1, 0), ActionXY(0.5, 0), ActionXY(0, 0), ActionXY(0.5, 0.5)]
    nav = moving(env.robot.get_full_state(), 0.8, 0.0)
    # a ship 3 m ahead coming straight at the robot: holding course and braking in
    # its path both end in contact, turning away does not
    oncoming = Human(env_cfg, 'humans')
    oncoming.set(8.0, 5.0, 0, 5, -1, 0, 0)
    forced, next_humans = emergency_candidates(policy, env, nav, [oncoming])
    assert policy.emergency_action(nav, forced, next_humans) == ActionXY(0.5, 0.5)
    held = [policy.held_clearance(nav, c, next_humans) for c in forced]
    assert held[3] > held[2] and held[3] > held[0]

    # an action that collides within this step loses to every one that does not,
    # however much room it would have afterwards
    forced[3] = forced[3]._replace(info=Collision(collision_kind='static'))
    assert policy.emergency_action(nav, forced, next_humans) != ActionXY(0.5, 0.5)

    # speeding up is allowed when it is what gets the ship clear: a ship crossing
    # from starboard aims at where the slow options will be
    crossing = Human(env_cfg, 'humans')
    crossing.set(5.6, 3.0, 5.6, 9, 0, 1, 0)
    forced, next_humans = emergency_candidates(policy, env, nav, [crossing])
    assert policy.emergency_action(nav, forced, next_humans) == ActionXY(1, 0)

    # end to end: a ship bearing down on the robot makes stopping unsafe
    oncoming = Human(env_cfg, 'humans')
    oncoming.set(6.0, 5.0, 0, 5, -3, 0, 0)
    oncoming.set_policy(FixedPolicy(ActionXY(-3, 0)))
    oncoming.time_step = env.time_step
    env.humans, env.human_times = [oncoming], [0]
    env.human_paths, env.human_wp_idx, env.human_wp_dir = [[]], [0], [1]
    policy.action_space = None
    env.robot.set(5, 5, 18, 5, 0, 0, 0)
    ob = env.observe_at(env.robot.get_position())
    action = env.robot.act(ob)
    assert policy.emergency
    assert action != ActionXY(0, 0), 'stopping is not assumed safe'
    nav = env.robot.get_full_state()
    humans, next_humans = env.predict_humans(JointState(nav, ob))
    stop = policy.check_action(0, nav, ActionXY(0, 0), humans, next_humans)
    chosen = policy.check_action(0, nav, action, humans, next_humans)
    assert (policy.held_clearance(nav, stop, next_humans)
            < policy.held_clearance(nav, chosen, next_humans))


def test_continuity_cost_breaks_near_ties(make_env, make_sarl):
    env = make_env()
    policy = make_sarl(env)
    policy.action_space = [ActionXY(0.5, 0), ActionXY(0, 0.5)]
    env.robot.set_velocity((0.5, 0.0))
    ob = env.observe_at(env.robot.get_position())
    policy.continuity_weight = 0.0
    values = []
    for _ in range(2):
        env.robot.act(ob)
        values.append(list(policy.action_values))
    assert values[0] == values[1]
    gap = abs(values[0][0] - values[0][1])
    # a weight large enough to outweigh the value gap keeps the current heading
    policy.continuity_weight = gap / (math.pi / 2) + 1e-3
    assert env.robot.act(ob) == ActionXY(0.5, 0)
    assert policy.action_values == pytest.approx(values[0]), 'reported values exclude the penalty'


def test_continuity_cost_never_holds_the_ship_back(make_env, make_sarl):
    env = make_env()
    policy = make_sarl(env)
    policy.continuity_weight = 1e6
    # at rest every direction is free, so the ship sets off on value alone
    policy.action_space = [ActionXY(0.5, 0), ActionXY(0, 0.5), ActionXY(0, 0)]
    ob = env.observe_at(env.robot.get_position())
    action = env.robot.act(ob)
    assert action == policy.candidates[int(np.argmax(policy.action_values))]
    assert action != ActionXY(0, 0)
    # speeding up or slowing down on the current course, and stopping, cost nothing
    env.robot.set_velocity((0.5, 0.0))
    policy.action_space = [ActionXY(0.6, 0), ActionXY(0.4, 0), ActionXY(0, 0), ActionXY(0.5, 0.1)]
    action = env.robot.act(ob)
    assert action != ActionXY(0.5, 0.1)
    free = [v for a, v in zip(policy.candidates, policy.action_values) if a.vy == 0]
    assert policy.action_values[policy.candidates.index(action)] == max(free)


def test_reaching_a_local_goal_does_not_force_a_stop(make_env, make_sarl):
    env = make_env()
    policy = make_sarl(env)
    env.robot.gx, env.robot.gy = env.robot.px + 0.1, env.robot.py
    policy.action_space = [ActionXY(1, 0)]
    assert env.robot.act(env.observe_at(env.robot.get_position())) == ActionXY(1, 0)


@pytest.mark.parametrize('query', [False, True])
def test_prediction_matches_execution_for_candidates(make_env, make_sarl, query):
    env = make_env(dynamic=[dict(px=8, py=5, gx=8, gy=10, path=[(8, 5), (8, 10)])])
    policy = make_sarl(env, query=query)
    ob = env.observe_at(env.robot.get_position())
    state = JointState(env.robot.get_full_state(), ob)
    nav = state.self_state
    humans, next_humans = env.predict_humans(state, query)
    for action in policy.candidate_actions(nav)[:5]:
        candidate = policy.check_action(0, nav, action, humans, next_humans)
        expected = env.onestep_lookahead(action) if query else env.predict_linear(state, action)
        assert (candidate.reward, candidate.done) == expected[1:3]
        assert vars(candidate.info) == vars(expected[3])


def test_motion_config_and_validation(make_env, make_sarl):
    import configparser
    from crowd_nav.policy.sarl import SARL
    from pathlib import Path
    cfg = configparser.RawConfigParser()
    cfg.read(Path(__file__).resolve().parents[1] / 'crowd_nav/configs/policy.config')
    policy = SARL()
    policy.configure(cfg)
    assert policy.constrained and policy.max_turn_rate == pytest.approx(math.radians(60))
    assert policy.safety_margin_static == 0.2 and policy.braking_check
    assert policy.safety_horizon == 3.0
    cfg.set('safety', 'safety_horizon', '-1')
    with pytest.raises(ValueError):
        SARL().configure(cfg)
    cfg.set('safety', 'safety_horizon', '3.0')
    cfg.set('motion', 'max_decel', '0')
    with pytest.raises(ValueError):
        SARL().configure(cfg)
    cfg.remove_section('motion')
    cfg.remove_section('safety')
    fresh = SARL()
    fresh.configure(cfg)
    assert fresh.max_accel == 0.5 and fresh.constrained


def test_training_phase_explores_safe_candidates(make_env, make_sarl):
    env = make_env(obstacles=[dict(cx=6.75, cy=5, w=1, h=1)])
    policy = make_sarl(env)
    policy.set_phase('train')
    policy.set_epsilon(1.0)
    policy.action_space = [ActionXY(1, 0), ActionXY(0, 1), ActionXY(0, 0)]
    policy.safety_margin_static = 0.5
    ob = env.observe_at(env.robot.get_position())
    for _ in range(10):
        assert env.robot.act(ob) != ActionXY(1, 0)
    assert policy.last_state is not None


def test_braking_land_check_matches_sweeping_every_substep(make_env, make_sarl):
    from crowd_sim.envs.utils.collision import static_clearance
    env = make_env(obstacles=[dict(cx=x + 0.5, cy=y + 0.5, w=1, h=1)
                              for x, y in [(7, 4), (7, 5), (3, 7), (8, 8), (2, 2)]])
    policy = make_sarl(env)
    rng = np.random.RandomState(0)
    for _ in range(300):
        nav = env.robot.get_full_state()
        nav.px, nav.py = rng.uniform(1, 9, 2)
        nav.position = (nav.px, nav.py)
        moving(nav, *rng.uniform(-1, 1, 2))
        # reference: the old per-substep sweep
        dt, speed = policy.time_step, math.hypot(nav.vx, nav.vy)
        heading, position = np.array((nav.vx, nav.vy)) / speed, np.array(nav.position)
        expected = math.inf
        while speed > 0:
            next_speed = max(0.0, speed - policy.max_decel * dt)
            end = position + heading * (speed + next_speed) / 2 * dt
            expected = min(expected, static_clearance(position, end, nav.radius, env.land_rects,
                                                      env.map_bounds) - policy.safety_margin_static)
            position, speed = end, next_speed
        got = policy.braking_clearance(nav, [])
        assert got == pytest.approx(expected, abs=1e-9)


def test_ship_inside_the_land_margin_may_hold_or_open_its_gap(make_env, make_sarl):
    # land from x = 6.0 and the robot 0.1 m from it, inside the 0.2 m margin:
    # every sweep starts too close, so before the coast margin nothing passed
    env = make_env(obstacles=[dict(cx=6.5, cy=5, w=1, h=1)])
    policy = make_sarl(env)
    env.robot.set(5.6, 5, 18, 5, 0, 0, 0)
    away, towards, along, stop = ActionXY(-0.5, 0), ActionXY(0.5, 0), ActionXY(0, 0.5), ActionXY(0, 0)
    policy.action_space = [away, towards, along, stop]
    action = env.robot.act(env.observe_at(env.robot.get_position()))
    assert not policy.emergency
    assert policy.safe_mask == [True, False, True, True], 'closing the gap still fails'
    assert action != towards
    assert policy.step_static_margin is None, 'the step margin does not outlive predict'
    # with room to spare the configured margin applies unchanged
    env.robot.set(3.0, 5, 18, 5, 0, 0, 0)
    assert policy.coast_margin(env.robot.get_full_state()) == policy.safety_margin_static
