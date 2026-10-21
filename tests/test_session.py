"""The Qt-free scenario session: map, ship, route, obstacles, episode, export."""
import pytest

from conftest import ROOT, FixedPolicy, write_map
from crowd_nav import evaluate
from crowd_nav.session import STATES, EpisodeResult, ScenarioError, ScenarioSession

MODEL_DIR = ROOT / 'crowd_nav' / 'data' / 'output-trained'
OUTCOMES = {'timeout', 'success', 'collision_dynamic', 'collision_static', 'collision_both'}


@pytest.fixture
def session(tmp_path):
    """A session with an injected no-op policy, so no torch and no weights are needed."""
    return ScenarioSession(str(MODEL_DIR), policy=FixedPolicy())


def load(session, tmp_path, **kwargs):
    session.load_map(write_map(tmp_path / 'map.png', **kwargs))
    return session


def plan(session, tmp_path, start=(5.5, 5.5), goal=(44.5, 44.5), **kwargs):
    load(session, tmp_path, **kwargs)
    session.set_start(start)
    session.set_goal(goal)
    session.plan_route()
    return session


# ------------------------------------------------------------------ orientation

def test_image_row_zero_is_the_top_of_the_world(session, tmp_path):
    load(session, tmp_path, blocks=[(45, 0, 50, 5)])      # top-right island
    assert session.land_mask.shape == (50, 50)
    assert session.land_mask[45, 45]                       # world top-right
    assert not session.land_mask[45, 5]                    # world top-left
    assert session.sim_grid.is_on_land((45.5, 45.5))
    assert not session.sim_grid.is_on_land((45.5, 5.5))
    # the belt is exactly the planning grid minus the simulated geometry
    assert session.margin_mask[44, 45] and not session.land_mask[44, 45]
    assert not session.margin_mask[45, 45]                 # real land is not "belt"


def test_state_walks_the_documented_ladder(session, tmp_path):
    assert session.state == 'no_map'
    load(session, tmp_path)
    assert session.state == 'map'
    assert session.time_step == 0.25
    session.set_start((12.3, 40.7))
    assert session.state == 'start_set'
    session.set_goal((40.5, 10.5))
    assert session.state == 'goal_set'
    session.plan_route()
    assert session.state == 'planned'
    session.generate_obstacles(2, 0)
    assert session.state == 'obstacles'
    session.run_episode(max_steps=3)
    assert session.state == 'ran'
    assert session.state in STATES


# ------------------------------------------------------------------- clicking

def test_world_from_click_and_snap(session, tmp_path):
    load(session, tmp_path)
    assert ScenarioSession.world_from_click(12.3, 40.7) == (12.3, 40.7)
    assert ScenarioSession.world_from_click(None, 1.0) is None
    assert ScenarioSession.world_from_click(1.0, None) is None
    assert ScenarioSession.world_from_click(-0.5, 10.0) is None
    assert ScenarioSession.world_from_click(10.0, 50.5) is None
    assert session.snap((12.3, 40.7)) == ((12, 40), (12.5, 40.5))


def test_set_start_rejects_land_and_the_safety_belt(session, tmp_path):
    load(session, tmp_path, blocks=[(45, 0, 50, 5)])
    with pytest.raises(ScenarioError, match='safety belt'):
        session.set_start((45.5, 45.5))        # the island itself
    with pytest.raises(ScenarioError, match='safety belt'):
        session.set_start((44.5, 45.5))        # one cell off it: the inflated belt
    with pytest.raises(ScenarioError):
        session.set_start(None)
    assert session.start_cell is None

    cell, centre = session.set_start((12.3, 40.7))
    assert (cell, centre) == ((12, 40), (12.5, 40.5))
    assert session.start_centre == (12.5, 40.5)
    assert session.state == 'start_set'


# ---------------------------------------------------------------------- route

def test_plan_route_stores_a_goal_first_path_and_a_start_first_route(session, tmp_path):
    plan(session, tmp_path, start=(12.3, 40.7), goal=(40.5, 10.5))
    scenario = session.scenario
    assert scenario['path'][0] == session.goal_cell == (40, 10)
    assert scenario['path'][-1] == session.start_cell == (12, 40)
    assert session.route[0] == session.start_centre == (12.5, 40.5)
    assert session.route[-1] == session.goal_centre
    assert scenario['cost'] > 0
    assert scenario['dyn'] == []
    assert session.route_length() == pytest.approx(41.0, abs=2.0)


def test_plan_route_rejects_one_cell(session, tmp_path):
    load(session, tmp_path)
    session.set_start((12.3, 40.7))
    session.set_goal((12.6, 40.2))
    assert session.start_cell == session.goal_cell
    with pytest.raises(ScenarioError, match='same cell'):
        session.plan_route()


def test_plan_route_reports_no_path_across_a_wall(session, tmp_path):
    load(session, tmp_path, blocks=[(25, 0, 26, 50)])
    session.set_start((10.5, 25.5))
    session.set_goal((40.5, 25.5))
    with pytest.raises(ScenarioError, match='No route'):
        session.plan_route()
    assert session.route is None and session.scenario is None


# ------------------------------------------------------------------ obstacles

def test_generate_obstacles_is_reproducible_and_keeps_off_land(session, tmp_path):
    plan(session, tmp_path)
    first = session.generate_obstacles(4, 0)
    assert len(first) == 4
    second = session.generate_obstacles(4, 0)
    assert [(o['px'], o['py'], o['gx'], o['gy']) for o in first] == \
           [(o['px'], o['py'], o['gx'], o['gy']) for o in second]
    assert session.state == 'obstacles'
    for obstacle in first:
        assert obstacle['path']            # each one got a land free route
        assert not session.plan_grid.is_on_land((obstacle['px'], obstacle['py']))
        assert not session.plan_grid.is_on_land((obstacle['gx'], obstacle['gy']))
    assert session.obstacle_table() == [(o['px'], o['py'], o['gx'], o['gy']) for o in first]


def test_generate_obstacles_changes_with_the_seed(session, tmp_path):
    plan(session, tmp_path)
    a = session.generate_obstacles(4, 0)
    b = session.generate_obstacles(4, 1)
    assert [(o['px'], o['py']) for o in a] != [(o['px'], o['py']) for o in b]


def test_zero_obstacles_is_allowed(session, tmp_path):
    plan(session, tmp_path)
    assert session.generate_obstacles(0, 0) == []
    assert session.scenario['dyn'] == []
    assert session.state == 'planned'


def test_a_corridor_without_room_asks_for_another_seed(session, tmp_path):
    # one free row survives the 1 m inflation, which is a route but no room to cross it
    load(session, tmp_path, white_rows=range(23, 26))
    session.set_start((5.5, 25.5))
    session.set_goal((44.5, 25.5))
    session.plan_route()
    with pytest.raises(ScenarioError, match='another seed'):
        session.generate_obstacles(3, 0)


def test_obstacles_need_a_route_first(session, tmp_path):
    load(session, tmp_path)
    session.set_start((5.5, 5.5))
    session.set_goal((44.5, 44.5))
    with pytest.raises(ScenarioError, match='Plan a route'):
        session.generate_obstacles(2, 0)


# ---------------------------------------------------------------------- run

def test_run_episode_records_every_step(session, tmp_path):
    plan(session, tmp_path)
    seen = []
    result = session.run_episode(max_steps=5, on_step=lambda k, n: seen.append((k, n)))
    assert isinstance(result, EpisodeResult)
    assert seen == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]
    assert result.metrics['steps'] == len(result.states) == 5
    assert result.metrics['outcome'] in OUTCOMES
    assert result.render_env() is session.env
    assert session.state == 'ran'


def test_export_results_csv_is_one_row_in_the_batch_columns(session, tmp_path):
    import csv
    plan(session, tmp_path)
    session.generate_obstacles(2, 7)
    result = session.run_episode(max_steps=4)
    path = session.export_results(result, tmp_path / 'run.csv')
    with open(path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    row = rows[0]
    assert list(row) == evaluate.CSV_FIELDS + ScenarioSession.CSV_EXTRA_FIELDS
    assert (row['map'], row['seed'], row['requested_n_dynamic']) == ('map', '7', '2')
    assert row['outcome'] == result.metrics['outcome'] and row['steps'] == '4'
    assert row['n_dynamic'] == str(len(session.scenario['dyn']))


def test_export_results_json_has_the_route_ships_and_trajectory(session, tmp_path):
    import json
    plan(session, tmp_path)
    session.generate_obstacles(2, 7)
    result = session.run_episode(max_steps=4)
    obstacles = session.obstacle_table()
    session.set_goal((40.5, 10.5))          # the window moves on; the export does not
    with open(session.export_results(result, tmp_path / 'run.json')) as fh:
        record = json.load(fh)
    assert record['metrics'] == json.loads(json.dumps(result.metrics))
    assert record['settings']['seed'] == 7 and record['settings']['goal'] == [44.5, 44.5]
    assert len(record['obstacles']) == len(obstacles)
    assert record['route'][0] == [5.5, 5.5] and record['route'][-1] == [44.5, 44.5]
    trajectory = record['trajectory']
    assert len(trajectory) == len(result.states) == 4
    assert [step['t'] for step in trajectory] == [0.0, 0.25, 0.5, 0.75]
    assert trajectory[0]['own_ship'][:2] == [5.5, 5.5]
    assert all(len(step['ships']) == len(obstacles) for step in trajectory)


def test_export_results_rejects_other_file_types(session, tmp_path):
    plan(session, tmp_path)
    result = session.run_episode(max_steps=2)
    with pytest.raises(ScenarioError, match='.csv or .json'):
        session.export_results(result, tmp_path / 'run.txt')
    with pytest.raises(ScenarioError, match='Cannot write'):
        session.export_results(result, tmp_path / 'missing' / 'run.csv')


def test_run_episode_propagates_cancellation(session, tmp_path):
    plan(session, tmp_path)

    def cancel(k, n):
        if k == 2:
            raise evaluate.EpisodeCancelled()

    with pytest.raises(evaluate.EpisodeCancelled):
        session.run_episode(max_steps=5, on_step=cancel)
    assert session.result is None


def test_run_episode_needs_a_route(session, tmp_path):
    load(session, tmp_path)
    with pytest.raises(ScenarioError, match='Plan a route'):
        session.run_episode(max_steps=2)


# ---------------------------------------------------------------------- state

def test_new_goal_clears_obstacles_and_result(session, tmp_path):
    plan(session, tmp_path)
    session.generate_obstacles(2, 0)
    session.run_episode(max_steps=5)
    assert session.state == 'ran' and session.result is not None

    session.set_goal((40.5, 40.5))
    assert session.state == 'goal_set'
    assert session.result is None
    assert session.scenario is None
    assert session.route is None
    assert session.obstacle_table() == []


def test_new_start_keeps_the_goal_but_drops_the_route(session, tmp_path):
    plan(session, tmp_path)
    session.generate_obstacles(2, 0)
    session.set_start((10.5, 10.5))
    assert session.goal_cell == (44, 44)          # the goal survives
    assert session.route is None and session.scenario is None
    assert session.state == 'goal_set'


# ----------------------------------------------------------------------- map

def test_load_map_rejects_unreadable_and_missing_files(session, tmp_path):
    broken = tmp_path / 'broken.png'
    broken.write_text('definitely not a png')
    with pytest.raises(ScenarioError, match='Cannot read map'):
        session.load_map(broken)
    with pytest.raises(ScenarioError, match='not found'):
        session.load_map(tmp_path / 'missing.png')
    assert session.state == 'no_map'
    assert session.plan_grid is None


def test_missing_model_config_is_reported_up_front(tmp_path):
    with pytest.raises(ScenarioError, match='env.config'):
        ScenarioSession(str(tmp_path), policy=FixedPolicy())


# ---------------------------------------------------------------------- slow

@pytest.mark.slow
def test_real_weights_run_a_short_episode(tmp_path):
    session = ScenarioSession(str(MODEL_DIR))          # the trained SARL policy
    plan(session, tmp_path)
    session.generate_obstacles(1, 0)
    result = session.run_episode(max_steps=3)
    assert result.metrics['outcome'] in OUTCOMES
    assert result.metrics['steps'] == len(result.states) == 3
