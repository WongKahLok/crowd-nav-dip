import csv
from types import SimpleNamespace

import pytest

from conftest import FixedPolicy
from crowd_nav import evaluate
from crowd_nav.utils.theta_star import HybridEnv
from crowd_sim.envs.utils.action import ActionXY


def run(make_env, obstacles=(), dynamic=(), action=ActionXY(0, 0), max_steps=2, on_step=None):
    env = make_env(obstacles=obstacles)
    env.robot.set_policy(FixedPolicy(action))
    scenario = dict(start=(5, 5), goal=(9, 5), path=[(9, 5), (5, 5)], cost=4,
                    dyn=list(dynamic))
    cfg = SimpleNamespace(max_steps=max_steps, no_global_plan=True,
                          local_goal_dist=1, monitor_dist=5)
    row = evaluate.run_episode(env, env.robot, HybridEnv(50, 50, 1), scenario, cfg,
                               on_step=on_step)
    row.update(map='test', seed=0, requested_n_dynamic=len(dynamic))
    return row, env


def test_time_and_complete_planned_path(make_env):
    row, env = run(make_env, action=ActionXY(1, 0), max_steps=120)
    assert row['outcome'] == 'success'
    assert row['planned_dist'] == 4
    assert row['sim_time'] == env.global_time == row['steps'] * env.time_step
    assert row['travelled_dist'] == row['steps'] * env.time_step
    assert row['detour_ratio'] == round(row['travelled_dist'] / 4, 3)
    assert row['min_sep_dynamic'] == ''
    assert row['metrics_version'] == 2


def test_timeout_and_budget_overrides_training_config(make_env):
    row, env = run(make_env, max_steps=120)
    assert row['outcome'] == 'timeout'
    assert env.global_time == row['sim_time'] == 30
    assert row['steps'] == 120
    assert env.time_limit == 30


@pytest.mark.parametrize('kind', ['dynamic', 'static', 'both'])
def test_collision_metrics_and_initial_overlap(make_env, kind):
    dyn = [dict(px=5.5, py=5.5, gx=8, gy=8)] if kind != 'static' else []
    boxes = [dict(cx=5.5, cy=5.5, w=1, h=1)] if kind != 'dynamic' else []
    row, env = run(make_env, obstacles=boxes, dynamic=dyn)
    assert row['outcome'] == 'collision_' + kind
    assert row['collision_kind'] == kind
    assert row['danger_frac'] == 1
    assert row['steps'] == 1 and row['sim_time'] == env.time_step
    if kind != 'static':
        assert row['min_sep_dynamic'] == -0.6
    if kind != 'dynamic':
        assert row['min_sep_static'] == -0.3
    assert row['n_dynamic'] == len(dyn)


def test_static_collision_recorded_between_safe_endpoints(make_env):
    row, env = run(make_env, obstacles=[dict(cx=7.5, cy=5.5, w=1, h=1)],
                   action=ActionXY(16, 0))
    assert row['outcome'] == 'collision_static'
    assert row['min_sep_static'] == -0.3
    assert row['danger_frac'] == 1
    assert env.robot.get_position() == (9.5, 5.5)
    assert not env.is_on_land(env.robot.get_position(), env.robot.radius)


def test_summary_failures_and_underfilled(make_env, capsys):
    row, _ = run(make_env)
    row['requested_n_dynamic'] = 2
    failed = evaluate.scenario_failed_row('test', 1, 2)
    evaluate.summarise([row, failed], 'test.csv')
    output = capsys.readouterr().out
    assert 'attempted=2 executed=1 scenario_failed=1 underfilled=1' in output
    assert 'test 0 1 0.000 0.000 0.000 0.000 0.000 1.000' in output
    assert failed['steps'] == failed['sim_time'] == failed['n_dynamic'] == ''


def test_empty_free_cells():
    assert evaluate.sample_scenario(None, [], 4, 20) is None


def test_on_step_sees_every_executed_step(make_env):
    seen = []
    row, _ = run(make_env, max_steps=5, on_step=lambda k, n: seen.append((k, n)))
    assert seen == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]
    assert row['steps'] == len(seen) == 5


def test_on_step_can_cancel_the_episode(make_env):
    def cancel(k, n):
        if k == 2:
            raise evaluate.EpisodeCancelled()

    with pytest.raises(evaluate.EpisodeCancelled):
        run(make_env, max_steps=5, on_step=cancel)


def test_build_env_wires_policy_geometry_and_robot(env_cfg, make_env, monkeypatch):
    env = make_env()
    grid = HybridEnv(50, 50, 1)
    grid.x_min = grid.y_min = 0
    grid.x_max = grid.y_max = 50
    policy = FixedPolicy()
    monkeypatch.setattr(evaluate.gym, 'make', lambda *args: env)
    built, robot = evaluate.build_env(env_cfg, SimpleNamespace(colregs=False), policy, grid)
    assert built is env
    assert env.robot is robot
    assert policy.env is env
    assert env.land_rects.shape == (len(grid.obstacles), 4)


def test_cli_records_generation_failure_and_actual_count(make_env, monkeypatch, tmp_path):
    env = make_env()
    grid = HybridEnv(50, 50, 1)
    grid.x_min = grid.y_min = 0
    grid.x_max = grid.y_max = 50
    scenario = dict(start=(5, 5), goal=(9, 5), path=[(9, 5), (5, 5)], cost=4, dyn=[])
    samples = iter([None, scenario])
    monkeypatch.setattr(evaluate, 'build_grids', lambda *args: (grid, grid))
    monkeypatch.setattr(evaluate, 'sample_scenario', lambda *args: next(samples))
    monkeypatch.setattr(evaluate, 'build_policy', lambda *args: FixedPolicy())
    monkeypatch.setattr(evaluate.gym, 'make', lambda *args: env)
    out = tmp_path / 'results.csv'
    monkeypatch.setattr('sys.argv', ['evaluate', '--env_config', 'crowd_nav/configs/env.config',
                                    '--policy', 'linear', '--static_map', 'test.png',
                                    '--episodes', '2', '--densities', '2', '--max_steps', '2',
                                    '--out', str(out)])
    evaluate.main()
    with out.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert rows[0]['outcome'] == 'scenario_failed'
    assert rows[0]['sim_time'] == ''
    assert rows[1]['n_dynamic'] == '0'
    assert rows[1]['requested_n_dynamic'] == '2'
    assert all(r['metrics_version'] == '2' for r in rows)
