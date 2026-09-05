import copy
import math

import numpy as np
import pytest
import torch

from conftest import FixedPolicy
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import Collision, ReachGoal, Timeout
from crowd_sim.envs.utils.state import JointState


@pytest.mark.parametrize('kind', ['train', 'map'])
def test_collision_uses_executed_velocity(make_env, env_cfg, kind):
    env = make_env(kind)
    human = Human(env_cfg, 'humans')
    human.set(6, 5, 0, 5, 0, 0, 0)
    human.set_policy(FixedPolicy(ActionXY(-4, 0)))
    human.time_step = env.time_step
    env.humans = [human]
    env.human_times = [0]
    if kind == 'map':
        env.human_paths, env.human_wp_idx, env.human_wp_dir = [[]], [0], [1]
    state = JointState(env.robot.get_full_state(), env.observe_at(env.robot.get_position()))
    assert not env.predict_linear(state, ActionXY(0, 0))[2]
    _, reward, done, info = env.step(ActionXY(0, 0))
    assert done and isinstance(info, Collision) and info.collision_kind == 'dynamic'
    assert reward == env.collision_penalty
    assert human.get_position() == (5, 5)


@pytest.mark.parametrize('kind', ['train', 'map'])
def test_constant_prediction_matches_execution(make_env, env_cfg, kind):
    env = make_env(kind)
    human = Human(env_cfg, 'humans')
    human.set(6, 5.7, 10, 5.7, -1, 0, 0)
    human.set_policy(FixedPolicy(ActionXY(-1, 0)))
    human.time_step = env.time_step
    env.humans, env.human_times = [human], [0]
    if kind == 'map':
        env.human_paths, env.human_wp_idx, env.human_wp_dir = [[]], [0], [1]
    action = ActionXY(1, 0)
    state = JointState(env.robot.get_full_state(), env.observe_at(env.robot.get_position()))
    predicted = env.predict_linear(state, action)
    actual = env.step(action)
    assert predicted[1:3] == actual[1:3]
    assert vars(predicted[3]) == vars(actual[3])
    assert [str(h) for h in predicted[0]] == [str(h) for h in actual[0]]


@pytest.mark.parametrize('kind', ['train', 'map'])
def test_episode_clock_and_terminal_priority(make_env, kind):
    env = make_env(kind)
    env.time_limit = 0.75
    for i in range(3):
        if kind == 'map':
            for _ in range(3):
                env.light_reset()
        assert env.global_time == i * 0.25
        env.onestep_lookahead(ActionXY(0, 0))
        assert env.global_time == i * 0.25
        _, _, done, info = env.step(ActionXY(0, 0))
        assert done == (i == 2)
    assert isinstance(info, Timeout) and env.global_time == 0.75


def test_local_goal_is_not_terminal(make_env):
    env = make_env()
    env.robot.gx, env.robot.gy = (5.25, 5)
    _, reward, done, _ = env.step(ActionXY(1, 0))
    assert reward == env.success_reward and not done
    env.robot_end = (5.5, 5)
    env.time_limit = 0.5
    _, _, done, info = env.step(ActionXY(1, 0))
    assert done and isinstance(info, ReachGoal)


def snapshot(env):
    return (env.global_time, str(env.robot.get_full_state()),
            [str(h.get_full_state()) for h in env.humans], list(env.human_wp_idx),
            list(env.human_wp_dir), copy.deepcopy(env.human_paths), len(env.states),
            list(env.human_times), [h.policy.sim for h in env.humans],
            [h.policy.last_state for h in env.humans])


def test_preview_is_pure_and_order_independent(make_env):
    dyn = [dict(px=8, py=5, gx=10, gy=8, path=[(8, 5), (8, 8), (10, 8)])]
    env = make_env(dynamic=dyn)
    human = env.humans[0]
    human.act([])
    cached = human.policy.sim
    cached_pos = cached.getAgentPosition(0)
    cached_vel = cached.getAgentVelocity(0)
    env.human_wp_idx[0] = 0
    before = snapshot(env)
    a, b = ActionXY(1, 0), ActionXY(0, 1)
    first = env.onestep_lookahead(a)
    env.onestep_lookahead(b)
    again = env.onestep_lookahead(a)
    assert snapshot(env) == before
    assert cached.getAgentPosition(0) == cached_pos
    assert cached.getAgentVelocity(0) == cached_vel
    assert [str(h) for h in first[0]] == [str(h) for h in again[0]]
    assert vars(first[3]) == vars(again[3])
    actual = env.step(a)
    assert first[1:3] == actual[1:3]
    assert [str(h) for h in first[0]] == [str(h) for h in actual[0]]
    assert vars(first[3]) == vars(actual[3])
    assert env.human_wp_idx[0] == 1


def test_static_geometry_replacement_and_boundary(make_env):
    box = dict(cx=1.5, cy=1.5, w=1, h=1)
    env = make_env(obstacles=[box])
    env.set_obstacles([box])
    assert len(env.humans_static) == 1
    env.set_obstacles([])
    assert not env.humans_static and env.land_rects.shape == (0, 4)
    env.robot.set(49.5, 5, 18, 5, 0, 0, 0)
    _, _, done, info = env.step(ActionXY(2, 0))
    assert done and info.collision_kind == 'static'


def test_water_guard_sweeps_whole_segment(make_env):
    env = make_env(obstacles=[dict(cx=7.5, cy=5.5, w=1, h=1)],
                   dynamic=[dict(px=6, py=5.5, gx=10, gy=5.5)])
    move = env.keep_action_on_water(env.humans[0], ActionXY(12, 0))
    assert move == ActionXY(0, 0)


@pytest.mark.parametrize('query', [False, True])
def test_prediction_static_observation_at_future_position(make_env, query):
    env = make_env(obstacles=[dict(cx=10.4, cy=5, w=1, h=1)])
    state = JointState(env.robot.get_full_state(), env.observe_at(env.robot.get_position()))
    assert not state.human_states
    action = ActionXY(4, 0)
    obs, _, _, info = (env.onestep_lookahead(action) if query else env.predict_linear(state, action))
    assert len(obs) == 1 and obs[0].is_static
    assert math.isfinite(info.min_sep_static)


@pytest.mark.parametrize('query', [False, True])
@pytest.mark.parametrize('static', [False, True])
def test_sarl_empty_and_static_only(make_env, make_sarl, query, static):
    env = make_env(obstacles=[dict(cx=7.5, cy=8.5, w=1, h=1)] if static else [])
    policy = make_sarl(env, query=query)
    ob = env.observe_at(env.robot.get_position())
    state = JointState(env.robot.get_full_state(), ob)
    tensor = policy.transform(state).unsqueeze(0)
    value = policy.model(tensor)
    assert torch.isfinite(value).all()
    if not static:
        assert torch.equal(tensor[:, :, 6:], torch.zeros_like(tensor[:, :, 6:]))
        expected = policy.model.mlp3(torch.cat([
            tensor[:, 0, :6], torch.zeros(1, 50)], dim=1))
        assert torch.allclose(value, expected)
        assert len(policy.get_attention_weights()) == 0
    action = env.robot.act(ob)
    assert action is not None and np.isfinite(policy.action_values).all()
    assert env.global_time == 0


@pytest.mark.parametrize('query', [False, True])
def test_terminal_actions_skip_network(make_env, make_sarl, query):
    env = make_env()
    policy = make_sarl(env, query=query)
    env.time_limit = env.time_step
    calls = []
    hook = policy.model.register_forward_hook(lambda *args: calls.append(True))
    env.robot.act(env.observe_at(env.robot.get_position()))
    hook.remove()
    assert not calls
    assert all(v == 0 for v in policy.action_values)


def test_om_empty_observation(make_env, make_sarl):
    env = make_env()
    policy = make_sarl(env, weights=False, with_om=True)
    tensor = policy.transform(JointState(env.robot.get_full_state(), [])).unsqueeze(0)
    assert tensor.shape == (1, 1, 61)
    assert torch.isfinite(policy.model(tensor)).all()


def test_linear_prediction_does_not_query_future(make_env, monkeypatch):
    env = make_env(dynamic=[dict(px=8, py=5, gx=10, gy=8)])
    state = JointState(env.robot.get_full_state(), env.observe_at(env.robot.get_position()))
    before = env.predict_linear(state, ActionXY(1, 0))
    def forbidden():
        raise AssertionError('Linear prediction must not calculate future human actions')
    monkeypatch.setattr(env, '_human_moves', forbidden)
    env.humans[0].gx, env.humans[0].gy = 0, 0
    env.humans[0].vx = 100
    after = env.predict_linear(state, ActionXY(1, 0))
    assert [str(h) for h in before[0]] == [str(h) for h in after[0]]
    assert vars(before[3]) == vars(after[3])


@pytest.mark.parametrize('query', [False, True])
@pytest.mark.parametrize('terminal', ['collision', 'goal'])
def test_terminal_geometry_skips_value(make_env, make_sarl, query, terminal):
    env = make_env(obstacles=[dict(cx=5.25, cy=5, w=0.1, h=0.1)]
                   if terminal == 'collision' else [])
    policy = make_sarl(env, query=query)
    policy.action_space = [ActionXY(1, 0)]
    if terminal == 'goal':
        env.robot_end = (5.25, 5)
    calls = []
    hook = policy.model.register_forward_hook(lambda *args: calls.append(True))
    env.robot.act(env.observe_at(env.robot.get_position()))
    hook.remove()
    assert not calls
    expected = env.collision_penalty if terminal == 'collision' else env.success_reward
    assert policy.action_values == [expected]


def test_guarded_human_action_is_collision_truth(make_env):
    env = make_env(obstacles=[dict(cx=5.7, cy=5.5, w=0.2, h=1)],
                   dynamic=[dict(px=6.2, py=5.5, gx=3, gy=5.5)])
    human = env.humans[0]
    human.vx = -4
    human.set_policy(FixedPolicy(ActionXY(-4, 0)))
    _, _, done, info = env.step(ActionXY(0, 0))
    assert not done and info.collision_kind is None
    assert human.get_position() == (6.2, 5.5)


@pytest.mark.parametrize('query', [False, True])
def test_swept_collision_outside_observation_radius(make_env, query):
    env = make_env(obstacles=[dict(cx=12, cy=5, w=1, h=1)])
    state = JointState(env.robot.get_full_state(), env.observe_at(env.robot.get_position()))
    assert not state.human_states
    action = ActionXY(100, 0)
    obs, _, done, info = (env.onestep_lookahead(action) if query else env.predict_linear(state, action))
    assert not obs
    assert done and info.collision_kind == 'static'


def test_multiple_orca_agents_preview_matches_repeated_execution(make_env):
    env = make_env(obstacles=[dict(cx=10.5, cy=9.5, w=1, h=1)], dynamic=[
        dict(px=8, py=5, gx=8, gy=10, path=[(8, 5), (8, 10)]),
        dict(px=6, py=8, gx=10, gy=8, path=[(6, 8), (10, 8)])])
    for _ in range(10):
        action = ActionXY(0.1, 0)
        before = snapshot(env)
        pred = env.onestep_lookahead(action)
        assert snapshot(env) == before
        actual = env.step(action)
        assert pred[1:3] == actual[1:3]
        assert vars(pred[3]) == vars(actual[3])
        for a, b in zip(pred[0], actual[0]):
            assert a.position == b.position and a.velocity == b.velocity


@pytest.mark.parametrize('query', [False, True])
def test_trained_sarl_runs_original_training_environment(make_env, make_sarl, query):
    env = make_env('train')
    make_sarl(env, query=query)
    obs = env.reset('test', 0)
    assert len(obs) == 5
    for i in range(3):
        action = env.robot.act(obs)
        assert np.isfinite(env.robot.policy.action_values).all()
        obs, _, done, _ = env.step(action)
        assert env.global_time == (i + 1) * env.time_step
        if done:
            break
