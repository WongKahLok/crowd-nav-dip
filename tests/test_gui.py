"""The Qt window: click handling and widget state only, no simulation."""
import os
import time
from types import SimpleNamespace

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')   # must precede QApplication

pytest.importorskip('PyQt5')

from PyQt5.QtWidgets import QApplication                          # noqa: E402

from conftest import ROOT, FixedPolicy, write_map                 # noqa: E402
from crowd_nav.gui import NavigatorApp                            # noqa: E402
from crowd_nav.session import ScenarioSession                     # noqa: E402

MODEL_DIR = ROOT / 'crowd_nav' / 'data' / 'output-trained'
SAME_AXES = object()


@pytest.fixture(scope='module')
def qapp():
    return QApplication.instance() or QApplication([])


def new_window(qapp, tmp_path, **map_kwargs):
    session = ScenarioSession(str(MODEL_DIR), policy=FixedPolicy())
    app = NavigatorApp(session, map_paths=[])
    assert app._load_map_file(str(write_map(tmp_path / 'map.png', **map_kwargs)))
    return app


@pytest.fixture
def app(qapp, tmp_path):
    window = new_window(qapp, tmp_path)
    yield window
    window.close()


def click(app, x, y, inaxes=SAME_AXES):
    axes = app.view.ax if inaxes is SAME_AXES else inaxes
    app._on_click(SimpleNamespace(inaxes=axes, button=1, xdata=x, ydata=y))


# ------------------------------------------------------------------- map

def test_window_starts_with_the_map_loaded_and_nothing_chosen(app):
    assert app.session.state == 'map'
    assert app.start_label.text() == 'Start: not set'
    assert app.goal_label.text() == 'Goal: not set'
    assert app.cost_label.text() == 'Route: not planned'
    assert 'map.png' in app.map_label.text()
    assert not app.plan_button.isEnabled()      # no points yet
    assert not app.run_button.isEnabled()
    assert not app.generate_button.isEnabled()


def test_playback_and_export_are_off_until_there_is_an_episode(app):
    assert not app.play_button.isEnabled()
    assert not app.frame_slider.isEnabled()
    assert not app.speed_combo.isEnabled()
    assert not app.export_button.isEnabled()
    assert not app.export_results_button.isEnabled()
    assert not app.cancel_button.isEnabled()


# ----------------------------------------------------------------- clicks

def test_two_clicks_set_the_points_and_plan_the_route(app):
    click(app, 12.3, 40.7)
    assert app.session.start_cell == (12, 40)
    assert app.session.start_centre == (12.5, 40.5)     # snapped to the cell centre
    assert '12.5' in app.start_label.text()
    assert app.goal_radio.isChecked()                   # auto advanced to the goal
    assert app.session.state == 'start_set'
    assert not app.plan_button.isEnabled()              # still only one point

    click(app, 40.5, 10.5)
    assert app.session.goal_cell == (40, 10)
    assert '40.5' in app.goal_label.text()
    assert app.session.state == 'planned'               # planning ran automatically
    assert app.cost_label.text().startswith('Route: cost')
    assert app.plan_button.isEnabled()
    assert app.run_button.isEnabled()
    assert app.generate_button.isEnabled()


def test_clicking_land_is_refused_and_changes_nothing(app):
    click(app, 0.5, 0.5)                                # the forced boundary ring
    assert app.session.start_cell is None
    assert app.session.state == 'map'
    assert app.start_label.text() == 'Start: not set'
    assert 'safety belt' in app.status.text()
    assert app.status.styleSheet()                      # red


def test_clicking_outside_the_map_is_refused(app):
    click(app, 60.0, 10.0)
    assert app.session.start_cell is None
    assert 'off the canvas' in app.status.text()


def test_clicks_outside_the_axes_and_right_clicks_are_ignored(app):
    click(app, 12.3, 40.7, inaxes=object())
    app._on_click(SimpleNamespace(inaxes=app.view.ax, button=3, xdata=12.3, ydata=40.7))
    assert app.session.start_cell is None


def test_re_setting_the_goal_drops_the_placed_obstacles(app):
    click(app, 5.5, 5.5)
    click(app, 44.5, 44.5)
    app.count_spin.setValue(2)
    app.generate_obstacles()
    assert app.session.state == 'obstacles'
    assert app.obstacle_table.rowCount() == 2

    click(app, 40.5, 40.5)                              # new goal -> replan
    assert app.session.state == 'planned'
    assert app.obstacle_table.rowCount() == 0
    assert app.placed_label.text() == 'Placed 0 / 2'


# -------------------------------------------------------------- obstacles

def test_generating_obstacles_fills_the_table(app):
    click(app, 5.5, 5.5)
    click(app, 44.5, 44.5)
    app.count_spin.setValue(3)
    app.seed_spin.setValue(0)
    app.generate_obstacles()
    assert app.session.state == 'obstacles'
    assert app.obstacle_table.rowCount() == 3
    assert app.placed_label.text() == 'Placed 3 / 3'
    assert app.run_button.isEnabled()
    assert app.obstacle_table.item(0, 0) is not None


def test_zero_obstacles_is_allowed_with_a_hint(app):
    click(app, 5.5, 5.5)
    click(app, 44.5, 44.5)
    app.count_spin.setValue(0)
    app.generate_obstacles()
    assert app.obstacle_table.rowCount() == 0
    assert app.session.state == 'planned'
    assert 'static map' in app.status.text()


def test_no_room_for_obstacles_asks_for_another_seed(qapp, tmp_path):
    window = new_window(qapp, tmp_path, white_rows=range(23, 26))
    try:
        click(window, 5.5, 25.5)
        click(window, 44.5, 25.5)
        window.count_spin.setValue(3)
        window.generate_obstacles()
        assert window.session.state == 'planned'
        assert 'seed' in window.status.text()
        assert window.status.styleSheet()
    finally:
        window.close()


# --------------------------------------------------------------- playback

def test_controls_lock_while_the_worker_runs(app, qapp):
    click(app, 5.5, 5.5)
    click(app, 44.5, 44.5)
    app.steps_spin.setValue(2000)          # the injected policy never moves, so it cannot end
    app.run_episode()
    # right after start() the QThread has not begun; the panel must already be locked
    assert app._is_running()
    assert not app.run_button.isEnabled()
    assert not app.load_button.isEnabled()
    assert not app.plan_button.isEnabled()
    assert not app.generate_button.isEnabled()
    assert not app.start_radio.isEnabled()
    assert app.cancel_button.isEnabled()

    before = app.session.start_centre
    click(app, 20.5, 20.5)                 # clicking while running must not move the ship
    assert app.session.start_centre == before
    assert 'Episode running' in app.status.text()

    app.cancel_episode()
    assert app.worker.wait(20000)
    for _ in range(100):                   # deliver the queued finished() signal
        qapp.processEvents()
        if not app._is_running():
            break
        time.sleep(0.01)
    assert not app._is_running()
    assert app.run_button.isEnabled()
    assert not app.cancel_button.isEnabled()
    assert app.session.result is None      # the partial episode was thrown away


def test_playback_buttons_activate_with_a_scene(app, qapp):
    click(app, 5.5, 5.5)
    click(app, 44.5, 44.5)
    app.session.run_episode(max_steps=3)                 # a real episode, no thread
    scene = app.view.show_episode(app.session.result)
    app.frame_slider.setRange(0, scene.frames - 1)
    assert scene.frames == 3
    app._sync_widgets()
    assert app.play_button.isEnabled()
    assert app.export_button.isEnabled()
    assert app.export_results_button.isEnabled()

    app._set_frame(2)
    assert app.view.scene.frame == 2
    assert 'frame 3/3' in app.time_label.text()
    app._on_slider(1)
    assert app.view.scene.frame == 1


def test_export_results_writes_the_file_and_reports_it(app, tmp_path):
    app.export_results(str(tmp_path / 'early.csv'))
    assert 'Nothing to export' in app.status.text()
    assert not (tmp_path / 'early.csv').exists()

    click(app, 5.5, 5.5)
    click(app, 44.5, 44.5)
    app.session.run_episode(max_steps=3)
    path = tmp_path / 'run.json'
    app.export_results(str(path))
    assert path.exists()
    assert 'Results written to' in app.status.text()
    app.export_results(str(tmp_path / 'run.txt'))
    assert '.csv or .json' in app.status.text()


def test_model_picker_lists_the_model_folders(app):
    names = [app.model_combo.itemText(i) for i in range(app.model_combo.count())]
    assert 'output-trained' in names
    assert app.model_combo.currentText() == 'output-trained'
    assert 'output-trained' in app.model_label.text()
    assert app.model_load_button.isEnabled()


def test_loading_another_model_keeps_the_map_and_clears_the_scenario(qapp, tmp_path):
    from conftest import fake_model
    other = fake_model(tmp_path / 'other-model')
    built = []

    def factory(model_dir):
        built.append(model_dir)
        return ScenarioSession(model_dir, policy=FixedPolicy())

    session = ScenarioSession(str(MODEL_DIR), policy=FixedPolicy())
    app = NavigatorApp(session, map_paths=[], model_root=str(tmp_path), session_factory=factory)
    try:
        assert app._load_map_file(str(write_map(tmp_path / 'map.png')))
        click(app, 5.5, 5.5)
        click(app, 44.5, 44.5)
        assert app.session.state == 'planned'

        assert app.load_model(str(MODEL_DIR)) and built == [], 'the loaded model is not rebuilt'
        app.model_combo.setCurrentIndex(app._model_index(str(other)))
        app.load_selected_model()
        assert built == [str(other)]
        assert app.session is not session and app.session.model_dir == str(other)
        assert app.session.state == 'map', 'same map, no start, goal or route'
        assert app.session.map_path == str(tmp_path / 'map.png')
        assert 'other-model' in app.model_label.text()
        assert 'set the start and goal again' in app.status.text()

        broken = fake_model(tmp_path / 'broken', weights=False)
        assert not app.load_model(str(broken))
        assert 'weights' in app.status.text()
        assert app.session.model_dir == str(other), 'a bad folder leaves the model alone'
    finally:
        app.close()
