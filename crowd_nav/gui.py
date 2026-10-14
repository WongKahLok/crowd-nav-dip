"""PyQt5 cockpit for building a scenario and watching SARL sail it.

    python -m crowd_nav.gui [--model_dir DIR] [--map PNG] [--gpu] [--colregs]

The window is one matplotlib axes that doubles as a click target and a playback screen,
a scrolling panel of controls on the right, and a log at the bottom. Everything that
touches the simulation goes through :class:`crowd_nav.session.ScenarioSession`; this
module only wires widgets to it, so a failed call is a message in the status bar rather
than a stack trace in the terminal.
"""
import argparse
import glob
import os
import sys
import threading
import time
import traceback

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from PIL import Image
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QFileDialog,
                             QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow,
                             QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
                             QRadioButton, QScrollArea, QSizePolicy, QSlider, QSpinBox,
                             QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from crowd_nav.session import ScenarioError, ScenarioSession

# this file is <repo>/crowd_nav/gui.py, so the repository root is two levels up
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_DIR = os.path.join(REPO_ROOT, 'crowd_nav', 'data', 'output-trained')
STATIC_MAP_DIR = os.path.join(REPO_ROOT, 'static_maps')

MAP_SIZE = 50.0
SPEEDS = [0.5, 1.0, 2.0, 4.0]
PLAY_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
               '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78']

LAND_CMAP = ListedColormap([(0.0, 0.0, 0.0, 0.0), (0.45, 0.45, 0.45, 1.0)])
MARGIN_CMAP = ListedColormap([(0.0, 0.0, 0.0, 0.0), (0.95, 0.55, 0.15, 0.35)])

class EpisodeWorker(QThread):
    """Runs one episode off the GUI thread so the window stays draggable."""

    progress = pyqtSignal(int, int)
    finished_ok = pyqtSignal(object)
    cancelled = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, session, max_steps, parent=None):
        super().__init__(parent)
        self.session = session
        self.max_steps = max_steps
        self._cancel = threading.Event()
        self._last_emit = 0.0

    def cancel(self):
        self._cancel.set()

    def _on_step(self, done, total):
        if self._cancel.is_set():
            # evaluate.run_episode lets this escape and drops the partial episode
            from crowd_nav import evaluate
            raise evaluate.EpisodeCancelled()
        now = time.monotonic()
        if done == total or done % 10 == 0 or now - self._last_emit > 0.1:
            self._last_emit = now
            self.progress.emit(done, total)

    def run(self):
        from crowd_nav import evaluate
        try:
            result = self.session.run_episode(self.max_steps, on_step=self._on_step)
        except evaluate.EpisodeCancelled:
            self.cancelled.emit()
        except Exception as exc:                      # surfaced in a dialog, not the console
            self.failed.emit('{}\n\n{}'.format(exc, traceback.format_exc()))
        else:
            self.finished_ok.emit(result)


class MapView(object):
    """Everything drawn on the single axes. Reads a session, never calls it."""

    def __init__(self, fig, ax, canvas):
        self.fig = fig
        self.ax = ax
        self.canvas = canvas
        self.session = None
        self.scene = None
        self.show_source = False

    # ------------------------------------------------------------------ base

    def draw_map(self, session):
        self.session = session
        self.scene = None
        ax = self.ax
        ax.clear()
        if self.show_source and session.map_path and os.path.exists(session.map_path):
            ax.imshow(Image.open(session.map_path), origin='upper', cmap='gray',
                      extent=(0, MAP_SIZE, 0, MAP_SIZE), alpha=0.35, zorder=0)
        ax.imshow(session.land_mask.astype(float), origin='lower', cmap=LAND_CMAP,
                  vmin=0, vmax=1, extent=(0, MAP_SIZE, 0, MAP_SIZE),
                  interpolation='nearest', zorder=1)
        ax.imshow(session.margin_mask.astype(float), origin='lower', cmap=MARGIN_CMAP,
                  vmin=0, vmax=1, extent=(0, MAP_SIZE, 0, MAP_SIZE),
                  interpolation='nearest', zorder=2)
        ax.set_xlim(0, MAP_SIZE)
        ax.set_ylim(0, MAP_SIZE)
        ax.set_aspect('equal')
        ax.set_xlabel('x(m)')
        ax.set_ylabel('y(m)')
        self.draw_markers(session)
        self.draw_route(session)
        self.draw_obstacles(session)
        self.canvas.draw_idle()

    def draw_markers(self, session):
        if session.start_centre is not None:
            self.ax.plot(*session.start_centre, marker='o', color='#2ca02c', markersize=9,
                         linestyle='None', zorder=6)
            self.ax.annotate('start ({:.1f}, {:.1f})'.format(*session.start_centre),
                             session.start_centre, textcoords='offset points',
                             xytext=(8, 8), fontsize=9, color='#2ca02c', zorder=7)
        if session.goal_centre is not None:
            self.ax.plot(*session.goal_centre, marker='*', color='#d62728', markersize=15,
                         linestyle='None', zorder=6)
            self.ax.annotate('goal ({:.1f}, {:.1f})'.format(*session.goal_centre),
                             session.goal_centre, textcoords='offset points',
                             xytext=(8, -14), fontsize=9, color='#d62728', zorder=7)

    def draw_route(self, session):
        if not session.route:
            return
        xs, ys = zip(*session.route)
        self.ax.plot(xs, ys, 'k--', linewidth=1.2, zorder=4)

    def draw_obstacles(self, session):
        obstacles = session.obstacle_table()
        for i, (px, py, gx, gy) in enumerate(obstacles):
            color = PLAY_COLORS[i % len(PLAY_COLORS)]
            path = session.scenario['dyn'][i].get('path') or []
            if path:
                xs, ys = zip(*path)
                self.ax.plot(xs, ys, '--', color=color, linewidth=0.8, alpha=0.7, zorder=3)
            self.ax.plot(px, py, marker='o', color=color, markersize=6, linestyle='None',
                         zorder=5)
            self.ax.plot(gx, gy, marker='x', color=color, markersize=7, linestyle='None',
                         zorder=5)

    # -------------------------------------------------------------- playback

    def show_episode(self, result):
        """Hand the axes over to the episode artists (land, route, ships, headings)."""
        from crowd_sim.envs.render import EpisodeAnimation
        self.ax.clear()
        self.scene = EpisodeAnimation(result.render_env(), self.ax)
        self.canvas.draw_idle()
        return self.scene

    def set_frame(self, k):
        if self.scene is None:
            return
        self.scene.update(k)
        self.canvas.draw_idle()


class NavigatorApp(QMainWindow):
    """The window: control panel on the right, map/playback on the left, log at the bottom."""

    def __init__(self, session, map_paths=(), parent=None):
        super().__init__(parent)
        self.session = session
        self.worker = None
        self._running = False
        self._closing = False
        self.mode = 'start'

        self.setWindowTitle('CrowdNav Scenario Builder')

        # ---- canvas -------------------------------------------------------
        self.fig = Figure(figsize=(7, 7))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.fig.add_subplot(111)
        self.view = MapView(self.fig, self.ax, self.canvas)
        self.canvas.mpl_connect('button_press_event', self._on_click)

        # ---- playback timer ----------------------------------------------
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self._build_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setFixedHeight(110)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(splitter, 1)
        layout.addWidget(self.log, 0)
        self.setCentralWidget(central)

        self.status = QLabel('Ready')
        self.statusBar().addWidget(self.status)

        self.map_combo.addItems([os.path.basename(p) for p in map_paths])
        self._map_paths = list(map_paths)
        self._sync_widgets()
        self._log('Model directory {}'.format(session.model_dir))

    # ----------------------------------------------------------- construction

    def _build_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(self._build_map_group())
        layout.addWidget(self._build_points_group())
        layout.addWidget(self._build_obstacles_group())
        layout.addWidget(self._build_run_group())
        layout.addWidget(self._build_playback_group())
        layout.addWidget(self._build_result_group())
        layout.addWidget(self._build_export_group())
        layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(340)
        return scroll

    def _build_map_group(self):
        group = QGroupBox('1. Map')
        layout = QVBoxLayout(group)
        self.map_combo = QComboBox()
        self.browse_button = QPushButton('Browse…')
        self.load_button = QPushButton('Load')
        self.browse_button.clicked.connect(self._browse_map)
        self.load_button.clicked.connect(self.load_selected_map)
        row = QHBoxLayout()
        row.addWidget(self.map_combo, 1)
        row.addWidget(self.browse_button)
        row.addWidget(self.load_button)
        layout.addLayout(row)
        self.show_source_box = QCheckBox('Overlay source image')
        self.show_source_box.toggled.connect(self._toggle_source)
        layout.addWidget(self.show_source_box)
        self.map_label = QLabel('No map loaded')
        self.map_label.setWordWrap(True)
        layout.addWidget(self.map_label)
        return group

    def _build_points_group(self):
        group = QGroupBox('2. Ship start / goal')
        layout = QVBoxLayout(group)
        self.start_radio = QRadioButton('Set start')
        self.goal_radio = QRadioButton('Set goal')
        self.start_radio.setChecked(True)
        self.point_group = QButtonGroup(group)
        self.point_group.addButton(self.start_radio)
        self.point_group.addButton(self.goal_radio)
        self.start_radio.toggled.connect(self._on_mode_change)
        row = QHBoxLayout()
        row.addWidget(self.start_radio)
        row.addWidget(self.goal_radio)
        layout.addLayout(row)
        self.start_label = QLabel('Start: not set')
        self.goal_label = QLabel('Goal: not set')
        layout.addWidget(self.start_label)
        layout.addWidget(self.goal_label)
        self.plan_button = QPushButton('Plan route')
        self.plan_button.clicked.connect(self.plan_route)
        layout.addWidget(self.plan_button)
        self.cost_label = QLabel('Route: not planned')
        layout.addWidget(self.cost_label)
        return group

    def _build_obstacles_group(self):
        group = QGroupBox('3. Obstacle ships')
        layout = QVBoxLayout(group)
        form = QFormLayout()
        self.count_spin = QSpinBox()
        self.count_spin.setRange(0, 20)
        self.count_spin.setValue(4)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2 ** 31 - 1)
        self.seed_spin.setValue(0)
        form.addRow('Count', self.count_spin)
        form.addRow('seed', self.seed_spin)
        layout.addLayout(form)
        self.generate_button = QPushButton('Generate random')
        self.generate_button.clicked.connect(self.generate_obstacles)
        layout.addWidget(self.generate_button)
        self.placed_label = QLabel('Placed 0 / 0')
        layout.addWidget(self.placed_label)
        self.obstacle_table = QTableWidget(0, 4)
        self.obstacle_table.setHorizontalHeaderLabels(['px', 'py', 'gx', 'gy'])
        self.obstacle_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.obstacle_table.setMaximumHeight(150)
        layout.addWidget(self.obstacle_table)
        return group

    def _build_run_group(self):
        group = QGroupBox('4. Run')
        layout = QVBoxLayout(group)
        form = QFormLayout()
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 200000)
        self.steps_spin.setValue(2000)
        form.addRow('max_steps', self.steps_spin)
        layout.addLayout(form)
        self.gpu_box = QCheckBox('Use GPU')
        layout.addWidget(self.gpu_box)
        row = QHBoxLayout()
        self.run_button = QPushButton('Run')
        self.cancel_button = QPushButton('Cancel')
        self.run_button.clicked.connect(self.run_episode)
        self.cancel_button.clicked.connect(self.cancel_episode)
        row.addWidget(self.run_button)
        row.addWidget(self.cancel_button)
        layout.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        return group

    def _build_playback_group(self):
        group = QGroupBox('5. Playback')
        layout = QVBoxLayout(group)
        row = QHBoxLayout()
        self.play_button = QPushButton('Play')
        self.play_button.clicked.connect(self.toggle_playback)
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(['{}×'.format(s) for s in SPEEDS])
        self.speed_combo.setCurrentIndex(1)
        self.speed_combo.currentIndexChanged.connect(self._on_speed_change)
        row.addWidget(self.play_button)
        row.addWidget(QLabel('Speed'))
        row.addWidget(self.speed_combo)
        layout.addLayout(row)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self.frame_slider)
        self.time_label = QLabel('t = 0.00 s')
        layout.addWidget(self.time_label)
        return group

    def _build_result_group(self):
        group = QGroupBox('6. Results')
        layout = QFormLayout(group)
        self.result_labels = {}
        for key, title in [('outcome', 'outcome'), ('steps', 'steps'), ('sim_time', 'sim_time'),
                           ('travelled_dist', 'travelled'), ('planned_dist', 'planned'),
                           ('detour_ratio', 'detour'), ('min_sep_dynamic', 'min_sep_dynamic'),
                           ('min_sep_static', 'min_sep_static'),
                           ('danger_frac', 'danger_frac'), ('emergency_frac', 'emergency_frac')]:
            label = QLabel('—')
            self.result_labels[key] = label
            layout.addRow(title, label)
        return group

    def _build_export_group(self):
        group = QGroupBox('7. Export')
        layout = QVBoxLayout(group)
        self.export_button = QPushButton('Export MP4…')
        self.export_button.clicked.connect(self.export_video)
        layout.addWidget(self.export_button)
        return group

    # --------------------------------------------------------------- helpers

    def _log(self, message):
        self.log.appendPlainText(message)

    def _set_status(self, message, error=False):
        self.status.setText(message)
        self.status.setStyleSheet('color: #b00020;' if error else '')
        if error:
            self._log('Error: {}'.format(message))

    def _fail(self, error):
        self._set_status(str(error), error=True)

    def _is_running(self):
        # an explicit flag, not QThread.isRunning(): right after start() the thread has not
        # begun yet and isRunning() is still False, which would leave the panel live.
        return self._running

    def _sync_widgets(self):
        state = self.session.state
        running = self._is_running()
        has_route = self.session.route is not None
        both_points = self.session.start_cell is not None and self.session.goal_cell is not None

        self.load_button.setEnabled(not running)
        self.browse_button.setEnabled(not running)
        self.map_combo.setEnabled(not running)
        self.start_radio.setEnabled(not running and state != 'no_map')
        self.goal_radio.setEnabled(not running and state != 'no_map')
        self.plan_button.setEnabled(not running and both_points)
        self.generate_button.setEnabled(not running and has_route)
        self.count_spin.setEnabled(not running and has_route)
        self.seed_spin.setEnabled(not running and has_route)
        self.run_button.setEnabled(not running and has_route)
        self.cancel_button.setEnabled(running)
        self.steps_spin.setEnabled(not running)
        self.gpu_box.setEnabled(not running)
        self.export_button.setEnabled(not running and self.session.result is not None)
        playing = self.view.scene is not None
        self.play_button.setEnabled(playing)
        self.frame_slider.setEnabled(playing)
        self.speed_combo.setEnabled(playing)

    def _refresh_after_change(self):
        self.view.draw_map(self.session)
        self.start_label.setText('Start: {}'.format(
            'not set' if self.session.start_centre is None
            else '({:.1f}, {:.1f})'.format(*self.session.start_centre)))
        self.goal_label.setText('Goal: {}'.format(
            'not set' if self.session.goal_centre is None
            else '({:.1f}, {:.1f})'.format(*self.session.goal_centre)))
        if self.session.scenario is None:
            self.cost_label.setText('Route: not planned')
        else:
            self.cost_label.setText('Route: cost {:.2f}, {} legs'.format(
                self.session.scenario['cost'], len(self.session.scenario['path']) - 1))
        self._fill_obstacle_table()
        self._clear_results()
        self._sync_widgets()

    def _fill_obstacle_table(self):
        rows = self.session.obstacle_table()
        self.obstacle_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                self.obstacle_table.setItem(i, j, QTableWidgetItem('{:.2f}'.format(value)))
        requested = self.count_spin.value() if self.session.route is not None else 0
        self.placed_label.setText('Placed {} / {}'.format(len(rows), requested))
        for warning in self.session.warnings:
            self._set_status(warning, error=True)

    def _clear_results(self):
        for label in self.result_labels.values():
            label.setText('—')

    def _show_results(self, metrics):
        for key, label in self.result_labels.items():
            value = metrics.get(key, '')
            label.setText('—' if value == '' else str(value))

    # ------------------------------------------------------------------- map

    def _browse_map(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Choose a map PNG', STATIC_MAP_DIR,
                                              'PNG (*.png);;All files (*)')
        if not path:
            return
        if path not in self._map_paths:
            self._map_paths.append(path)
            self.map_combo.addItem(os.path.basename(path))
        self.map_combo.setCurrentIndex(self._map_paths.index(path))
        self.load_selected_map()

    def load_selected_map(self, *_):
        index = self.map_combo.currentIndex()
        if not (0 <= index < len(self._map_paths)):
            self._set_status('No map to load', error=True)
            return
        if self._load_map_file(self._map_paths[index]):
            self._log('Loaded map {}'.format(self._map_paths[index]))

    def _load_map_file(self, path):
        self._stop_playback()
        try:
            self.session.load_map(path)
        except ScenarioError as error:
            self._fail(error)
            return False
        self.map_label.setText('Map {} ({}x{})'.format(
            os.path.basename(path), self.session.land_mask.shape[1],
            self.session.land_mask.shape[0]))
        self.mode = 'start'
        self.start_radio.setChecked(True)
        self._refresh_after_change()
        self._set_status('Map loaded - click the canvas to set the start')
        return True

    def _toggle_source(self, checked):
        self.view.show_source = bool(checked)
        if self.session.plan_grid is not None:
            self.view.draw_map(self.session)

    # ---------------------------------------------------------------- points

    def _on_mode_change(self, *_):
        self.mode = 'start' if self.start_radio.isChecked() else 'goal'

    def _on_click(self, event):
        if getattr(event, 'inaxes', None) is not self.ax:
            return
        if getattr(event, 'button', 1) != 1:
            return
        if self._is_running():
            self._set_status('Episode running - cancel it or wait for it to finish', error=True)
            return
        if self.session.plan_grid is None:
            self._set_status('Load a map first', error=True)
            return
        world = ScenarioSession.world_from_click(event.xdata, event.ydata)
        if world is None:
            self._set_status('That point is off the canvas', error=True)
            return
        try:
            if self.mode == 'start':
                self.session.set_start(world)
                self._log('Start ({:.2f}, {:.2f}) -> cell {}'.format(
                    world[0], world[1], self.session.start_cell))
                self.goal_radio.setChecked(True)
            else:
                self.session.set_goal(world)
                self._log('Goal ({:.2f}, {:.2f}) -> cell {}'.format(
                    world[0], world[1], self.session.goal_cell))
        except ScenarioError as error:
            self._fail(error)
            return
        self._refresh_after_change()
        if (self.session.start_cell is not None and self.session.goal_cell is not None
                and self.session.scenario is None):
            self.plan_route()

    def plan_route(self):
        try:
            scenario = self.session.plan_route()
        except ScenarioError as error:
            self._fail(error)
            return
        self._refresh_after_change()
        self._set_status('Route planned, cost {:.2f}'.format(scenario['cost']))

    # ------------------------------------------------------------- obstacles

    def generate_obstacles(self):
        n = self.count_spin.value()
        seed = self.seed_spin.value()
        try:
            obstacles = self.session.generate_obstacles(n, seed)
        except ScenarioError as error:
            self._fail(error)
            return
        self._refresh_after_change()
        if n == 0:
            self._set_status('No obstacle ships - static map only')
        elif len(obstacles) < n:
            self._set_status('Placed {} / {}'.format(len(obstacles), n), error=True)
        else:
            self._set_status('Placed {} obstacle ships (seed {})'.format(len(obstacles), seed))

    # ------------------------------------------------------------------- run

    def run_episode(self):
        if self._is_running():
            return
        self._stop_playback()
        if self.gpu_box.isChecked():
            self._apply_device()
        self.view.scene = None
        self.worker = EpisodeWorker(self.session, self.steps_spin.value(), self)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.cancelled.connect(self._on_cancelled)
        self.worker.failed.connect(self._on_failed)
        # queued after run() returns, so the thread is really finished when the widgets sync
        self.worker.finished.connect(self._on_worker_done)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self._set_status('Running…')
        self._running = True
        self._sync_widgets()
        self.worker.start()

    def _apply_device(self):
        try:
            import torch
        except ImportError:
            torch = None
        if torch is None or not torch.cuda.is_available():
            self.gpu_box.setChecked(False)
            self._set_status('No CUDA available - falling back to CPU', error=True)
            return
        self.session.policy.set_device(torch.device('cuda:0'))

    def cancel_episode(self):
        if self.worker is not None:
            self.worker.cancel()
            self._set_status('Cancelling…')

    def _on_progress(self, done, total):
        if self._closing:
            return
        self.progress.setValue(int(100 * done / max(total, 1)))
        self._set_status('Running… {}/{} steps'.format(done, total))

    def _on_finished(self, result):
        if self._closing:
            return
        self.progress.setValue(100)
        scene = self.view.show_episode(result)
        self.frame_slider.setRange(0, scene.frames - 1)
        self.frame_slider.setValue(0)
        self._update_time_label(0)
        self._show_results(result.metrics)
        self._sync_widgets()
        self._set_status('Finished: {}, {} steps, travelled {}'.format(
            result.metrics['outcome'], result.metrics['steps'],
            result.metrics['travelled_dist']))

    def _on_cancelled(self):
        if self._closing:
            return
        self.progress.setValue(0)
        self._set_status('Cancelled - the partial episode was discarded')
        self._sync_widgets()

    def _on_failed(self, message):
        if self._closing:
            return
        self.progress.setValue(0)
        self._clear_results()
        self._fail(message.splitlines()[0])
        self._log(message)
        QMessageBox.critical(self, 'Episode failed', message)
        self._sync_widgets()

    def _on_worker_done(self):
        if self._closing:
            return
        self._running = False
        self._sync_widgets()

    # -------------------------------------------------------------- playback

    def _interval_ms(self):
        speed = SPEEDS[max(self.speed_combo.currentIndex(), 0)]
        return max(int(self.session.time_step * 1000 / speed), 1)

    def _stop_playback(self):
        self.timer.stop()
        self.play_button.setText('Play')

    def toggle_playback(self):
        if self.view.scene is None:
            return
        if self.timer.isActive():
            self._stop_playback()
        else:
            self.timer.start(self._interval_ms())
            self.play_button.setText('Pause')

    def _on_speed_change(self, *_):
        if self.timer.isActive():
            self.timer.start(self._interval_ms())

    def _tick(self):
        scene = self.view.scene
        if scene is None:
            self._stop_playback()
            return
        nxt = scene.frame + 1
        if nxt >= scene.frames:
            nxt = 0
        self._set_frame(nxt)

    def _on_slider(self, value):
        if self.view.scene is not None and value != self.view.scene.frame:
            self._set_frame(value)

    def _set_frame(self, k):
        self.view.set_frame(k)
        self._update_time_label(k)

    def _update_time_label(self, k):
        scene = self.view.scene
        if scene is None:
            self.time_label.setText('t = 0.00 s')
            return
        self.time_label.setText('t = {:.2f} s   frame {}/{}'.format(
            k * self.session.time_step, k + 1, scene.frames))
        if self.frame_slider.value() != k:
            self.frame_slider.setValue(k)

    # ---------------------------------------------------------------- export

    def export_video(self):
        result = self.session.result
        if result is None:
            self._set_status('Nothing to export yet', error=True)
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Export video', 'episode.mp4',
                                              'MP4 (*.mp4);;All files (*)')
        if not path:
            return
        self._stop_playback()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.export_button.setEnabled(False)
        self.run_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)

        def report(done, total):
            self.progress.setValue(int(100 * done / max(total, 1)))
            QApplication.processEvents()

        try:
            written = self.session.export_video(result, path, progress=report)
        except Exception as error:
            self._log(traceback.format_exc())
            self._fail(error)
        else:
            self.progress.setValue(100)
            self._set_status('Video written to {}'.format(written))
            self._log('Video written to {}'.format(written))
        finally:
            QApplication.restoreOverrideCursor()
            self._sync_widgets()

    # --------------------------------------------------------------- closing

    def closeEvent(self, event):
        self._closing = True
        self.timer.stop()
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            if not self.worker.wait(5000):
                self._log('Worker thread did not exit within 5 seconds')
        event.accept()


def main(argv=None):
    parser = argparse.ArgumentParser('CrowdNav scenario builder GUI')
    parser.add_argument('--model_dir', default=None,
                        help='trained weights directory, default {}'.format(
                            os.path.relpath(DEFAULT_MODEL_DIR, REPO_ROOT)))
    parser.add_argument('--map', default=None, help='map PNG to load at startup')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--colregs', default=False, action='store_true',
                        help='sail by the COLREGs: SARL with the rule filter (sarl_colregs)')
    parser.add_argument('--safety_distance', type=int, default=1)
    args = parser.parse_args(argv)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    model_dir = args.model_dir or DEFAULT_MODEL_DIR
    try:
        session = ScenarioSession(model_dir, safety_distance=args.safety_distance,
                                  colregs=args.colregs,
                                  device='cuda:0' if args.gpu else 'cpu')
    except Exception as error:
        QMessageBox.critical(None, 'CrowdNav', 'Could not load the policy:\n{}\n\n{}'.format(
            error, traceback.format_exc()))
        return 1

    maps = sorted(glob.glob(os.path.join(STATIC_MAP_DIR, '*.png')))
    window = NavigatorApp(session, maps)
    if args.map:
        window._map_paths.insert(0, args.map)
        window.map_combo.insertItem(0, os.path.basename(args.map))
        window.map_combo.setCurrentIndex(0)
    if window._map_paths:
        window.load_selected_map()
    window.resize(1280, 860)
    window.show()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
