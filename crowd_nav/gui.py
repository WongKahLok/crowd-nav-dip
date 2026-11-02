"""PyQt5 cockpit for building a scenario and watching SARL sail it.

    python -m crowd_nav.gui [--model_dir DIR] [--map PNG] [--gpu] [--colregs]

``--colregs`` starts with SARL + COLREGs; the Policy choice in the window switches.

The window is one matplotlib axes that doubles as a click target and a playback screen,
a scrolling panel of controls on the right, and a log at the bottom. Everything that
touches the simulation goes through :class:`crowd_nav.session.ScenarioSession`; this
module only wires widgets to it, so a failed call is a message in the status bar rather
than a stack trace in the terminal.
"""
import argparse
import glob
import html
import os
import sys
import threading
import time
import traceback

# torch must load before PyQt5 on Windows, otherwise c10.dll fails with WinError 1114
import torch  # noqa: F401
import numpy as np
from matplotlib import rcParams
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.colors import ListedColormap, same_color
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PIL import Image
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPalette
from PyQt5.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                             QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMainWindow,
                             QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
                             QRadioButton, QScrollArea, QSizePolicy, QSlider, QSpinBox,
                             QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from crowd_nav.session import ScenarioError, ScenarioSession, find_model_dirs, model_dir_problem

# this file is <repo>/crowd_nav/gui.py, so the repository root is two levels up
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ROOT = os.path.join(REPO_ROOT, 'crowd_nav', 'data')
# the policies the window offers: policy_factory name, label
POLICIES = [('sarl', 'SARL'), ('sarl_colregs', 'SARL + COLREGs')]
DEFAULT_MODEL_DIR = os.path.join(MODEL_ROOT, 'output-sarl')
STATIC_MAP_DIR = os.path.join(REPO_ROOT, 'static_maps')

MAP_SIZE = 50.0
SPEEDS = [0.5, 1.0, 2.0, 4.0]
PLAY_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
               '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78']

# playback ship marker: a pointed hull facing +x, rotated to the heading every frame
SHIP_SHAPE = np.array([(1.0, 0.0), (-0.7, 0.6), (-0.35, 0.0), (-0.7, -0.6), (1.0, 0.0)])
OWN_SHIP_COLOR = '#ffd400'
OWN_SHIP_SIZE = 16
SHIP_SIZE = 11
TRAIL_FRAMES = 40           # 10 s of wake at the 0.25 s time step
PLAYBACK_BOTTOM = 0.2       # figure margin below the axes for the playback legend

# Results panel: how each metric is shown. Exports keep the full-precision values.
METRIC_FORMATS = {
    'sim_time': '{:.1f} s', 'travelled_dist': '{:.2f} m', 'planned_dist': '{:.2f} m',
    'detour_ratio': '{:.3f}×', 'min_sep_dynamic': '{:.2f} m', 'min_sep_static': '{:.2f} m',
}
PERCENT_METRICS = ('danger_frac', 'emergency_frac')

# Light and dark themes for the widgets, the log and the map canvas. 'ink' is the colour
# of axes, labels, the route and the time; 'water' is the map's background.
THEMES = {
    'light': {
        'figure': 'white', 'water': 'white', 'ink': 'black',
        'warning': '#b26a00', 'error': '#b00020', 'detail': '#777777', 'stamp': '#888888',
        'primary_disabled_bg': '#cfd8dc', 'primary_disabled_fg': '#eceff1',
    },
    'dark': {
        'figure': '#1e1e1e', 'water': '#14232e', 'ink': '#e0e0e0',
        'warning': '#ffb74d', 'error': '#ff6b6b', 'detail': '#9e9e9e', 'stamp': '#8a8a8a',
        'primary_disabled_bg': '#3a4a52', 'primary_disabled_fg': '#7d8f99',
    },
}
THEME_POLL_MS = 2000        # how often the window checks the Windows light/dark setting

# the buttons that move the scenario on to its next stage
PRIMARY_BUTTON_STYLE = """
QPushButton[primary="true"] {{
    background: #1565c0; color: white; font-weight: 600;
    border: none; border-radius: 4px; padding: 5px 10px;
}}
QPushButton[primary="true"]:hover {{ background: #1976d2; }}
QPushButton[primary="true"]:pressed {{ background: #0d47a1; }}
QPushButton[primary="true"]:disabled {{ background: {primary_disabled_bg}; color: {primary_disabled_fg}; }}
"""


def system_theme():
    """'dark' when Windows apps are set to dark mode, otherwise (and off Windows) 'light'."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r'Software\Microsoft\Windows\CurrentVersion\Themes\Personalize') as key:
            value, _ = winreg.QueryValueEx(key, 'AppsUseLightTheme')
    except (ImportError, OSError):
        return 'light'
    return 'light' if value else 'dark'


def dark_palette():
    """A dark palette for the Fusion style."""
    text, disabled = QColor('#e0e0e0'), QColor('#7a7a7a')
    palette = QPalette()
    for role, color in ((QPalette.Window, '#2b2b2b'), (QPalette.Base, '#1e1e1e'),
                        (QPalette.AlternateBase, '#2b2b2b'), (QPalette.Button, '#353535'),
                        (QPalette.ToolTipBase, '#3c3c3c'), (QPalette.Highlight, '#2a82da'),
                        (QPalette.Link, '#64b5f6'), (QPalette.BrightText, '#ff6b6b')):
        palette.setColor(role, QColor(color))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText, QPalette.ToolTipText):
        palette.setColor(role, text)
        palette.setColor(QPalette.Disabled, role, disabled)
    palette.setColor(QPalette.HighlightedText, QColor('white'))
    palette.setColor(QPalette.PlaceholderText, disabled)
    return palette


def format_metric(key, value):
    """One Results value as shown in the panel: rounded, with its unit."""
    if value == '' or value is None:
        return '—'
    try:
        if key in PERCENT_METRICS:
            return '{:.1f} %'.format(100 * float(value))
        if key in METRIC_FORMATS:
            return METRIC_FORMATS[key].format(float(value))
    except (TypeError, ValueError):
        pass
    return str(value)


# log and status bar: warnings are things to adjust or retry, errors are failures; their
# colours come from the theme
LOG_PREFIXES = {'warning': 'Warning: ', 'error': 'Error: '}


# One look on every platform: Qt's Fusion style, and one font for the widgets and the
# map canvas. Segoe UI ships with Windows; elsewhere Qt and matplotlib fall back to the
# system sans-serif.
UI_FONT = 'Segoe UI'
UI_FONT_SIZE = 9
PANEL_MIN_WIDTH = 400       # px; keeps the model and map names readable in their dropdowns


def apply_style(app):
    """Fusion style and a single font for the window and the matplotlib canvas."""
    app.setStyle('Fusion')
    font = QFont(UI_FONT, UI_FONT_SIZE)
    font.setStyleHint(QFont.SansSerif)
    app.setFont(font)
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = [UI_FONT, 'DejaVu Sans']


def fit_combo(combo, chars=14):
    """Let a file-name dropdown shrink to ``chars`` characters instead of growing to its
    longest name, which would widen the whole panel; the tooltip has the full path."""
    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(chars)


def set_item_tooltips(combo, paths):
    """Full path as each item's tooltip; the narrow panel cuts long file names short."""
    for i, path in enumerate(paths):
        combo.setItemData(i, os.path.abspath(path), Qt.ToolTipRole)


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
        self.theme = THEMES['light']

    def apply_theme(self):
        """Recolour the figure for ``self.theme``. Anything drawn in the ink of either theme
        (axes, labels, the route, the playback time) takes this theme's ink; coloured
        markers keep their colours."""
        theme, ax = self.theme, self.ax
        inks = [t['ink'] for t in THEMES.values()]

        def is_ink(color):
            return any(same_color(color, ink) for ink in inks)

        self.fig.set_facecolor(theme['figure'])
        ax.set_facecolor(theme['water'])
        for spine in ax.spines.values():
            spine.set_edgecolor(theme['ink'])
        ax.tick_params(colors=theme['ink'], which='both')
        ax.xaxis.label.set_color(theme['ink'])
        ax.yaxis.label.set_color(theme['ink'])
        for line in ax.lines:
            if is_ink(line.get_color()):
                line.set_color(theme['ink'])
        for text in ax.texts:
            if is_ink(text.get_color()):
                text.set_color(theme['ink'])
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_color(theme['ink'])
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ base

    def draw_map(self, session):
        self.session = session
        self.scene = None
        ax = self.ax
        ax.clear()
        self.fig.subplots_adjust(bottom=rcParams['figure.subplot.bottom'])
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
        self.apply_theme()

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
        self._style_ships()
        self._update_ships(0)
        self.apply_theme()
        return self.scene

    def set_frame(self, k):
        if self.scene is None:
            return
        self.scene.update(k)
        self._update_ships(k)
        self.canvas.draw_idle()

    def _style_ships(self):
        """Draw every ship as a heading-shaped marker with a short trail.

        The scene's circles are true to scale (0.3 m), a few pixels on a 50 m map, so
        they are hidden and fixed-size markers on top carry the position and heading.
        """
        scene, ax = self.scene, self.ax
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        scene.robot.set_visible(False)
        for human in scene.humans:
            human.set_visible(False)
        for ring in (scene.robot_outer, scene.robot_detector):
            ring.set_alpha(0.6)
        colors = [OWN_SHIP_COLOR] + [PLAY_COLORS[i % len(PLAY_COLORS)]
                                     for i in range(len(scene.humans))]
        self.trails = [ax.plot([], [], '-', color=c, linewidth=2.5 if i == 0 else 1.5,
                               alpha=0.6, solid_capstyle='round', zorder=7)[0]
                       for i, c in enumerate(colors)]
        self.ships = [ax.plot([], [], linestyle='None', marker=SHIP_SHAPE,
                              markersize=OWN_SHIP_SIZE if i == 0 else SHIP_SIZE,
                              markerfacecolor=c, markeredgecolor='black',
                              markeredgewidth=1.0 if i == 0 else 0.6, zorder=8)[0]
                      for i, c in enumerate(colors)]
        handles = [
            Line2D([], [], linestyle='None', marker='o', markersize=9,
                   markerfacecolor=OWN_SHIP_COLOR, markeredgecolor='black', label='Own ship'),
            Line2D([], [], linestyle='None', marker='o', markersize=7,
                   markerfacecolor=PLAY_COLORS[0], markeredgecolor='black',
                   label='Other ships'),
            Line2D([], [], linestyle='None', marker='*', markersize=11, color='red',
                   label='Goal'),
            Line2D([], [], linestyle='None', marker='o', markersize=5, color='m',
                   label='Steering target'),
            Line2D([], [], linestyle='--', color='m', alpha=0.6, label='Distance to target'),
            Line2D([], [], linestyle='--', color='orange', alpha=0.6,
                   label='Detection range'),
        ]
        ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3,
                  fontsize=9, frameon=False)
        self.fig.subplots_adjust(bottom=PLAYBACK_BOTTOM)      # room for the legend

    def _positions(self, agent):
        scene = self.scene
        if agent == 0:
            return scene.robot_positions
        return [frame[agent - 1] for frame in scene.human_positions]

    def _heading(self, agent, k):
        """Smoothed heading from the scene's arrows; from the motion when it has none."""
        scene = self.scene
        if agent < len(scene.orientations):
            (x0, y0), (x1, y1) = scene.orientations[agent][k]
        else:
            positions = self._positions(agent)
            (x0, y0) = positions[max(k - 1, 0)]
            (x1, y1) = positions[min(max(k, 1), len(positions) - 1)]
        return np.arctan2(y1 - y0, x1 - x0)

    def _update_ships(self, k):
        for agent, (ship, trail) in enumerate(zip(self.ships, self.trails)):
            positions = self._positions(agent)
            tail = positions[max(0, k - TRAIL_FRAMES):k + 1]
            trail.set_data([p[0] for p in tail], [p[1] for p in tail])
            ship.set_data([positions[k][0]], [positions[k][1]])
            theta = self._heading(agent, k)
            c, s = np.cos(theta), np.sin(theta)
            ship.set_marker(SHIP_SHAPE @ np.array([[c, s], [-s, c]]))


class NavigatorApp(QMainWindow):
    """The window: control panel on the right, map/playback on the left, log at the bottom."""

    def __init__(self, session, map_paths=(), parent=None, model_root=MODEL_ROOT,
                 session_factory=None, policy_builder=None, theme='system'):
        """``session_factory(model_dir)`` builds the session for another model folder;
        by default a ScenarioSession with the current session's settings.
        ``policy_builder(name)`` returns a ready policy for the Policy choice; by default
        the session loads it from its model folder. ``theme`` is 'system' (follow
        Windows), 'light' or 'dark'."""
        super().__init__(parent)
        self.session = session
        self._session_factory = session_factory or self._session_like_current
        self._policy_builder = policy_builder
        self.worker = None
        self._running = False
        self._closing = False
        self.mode = 'start'
        self._ship_start = None      # first click of a hand-placed obstacle ship

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

        # ---- theme: 'system' keeps following the Windows light/dark setting ----
        self._status_level = 'info'
        self._theme_timer = QTimer(self)
        self._theme_timer.timeout.connect(self._follow_system_theme)
        if theme == 'system':
            self.set_theme(system_theme())
            self._theme_timer.start(THEME_POLL_MS)
        else:
            self.set_theme(theme)

        self.map_combo.addItems([os.path.basename(p) for p in map_paths])
        self._map_paths = list(map_paths)
        set_item_tooltips(self.map_combo, self._map_paths)
        self._sync_widgets()
        self._model_paths = []
        for path in find_model_dirs(model_root) + [session.model_dir]:
            self._add_model_path(path)
        self.model_combo.setCurrentIndex(self._model_index(session.model_dir))
        self._show_model()
        self._log('Model directory {}'.format(session.model_dir))
        self._fit_panel()                   # now that the dropdowns hold their names

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
        for button in (self.plan_button, self.generate_button, self.run_button):
            button.setProperty('primary', True)
        self.panel = panel                  # restyled by set_theme
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.panel_scroll = scroll
        self._fit_panel()
        return scroll

    def _fit_panel(self):
        """Wide enough for the widest section next to the vertical scroll bar, so the panel
        only ever scrolls up and down. Rerun after a style change, which moves widths."""
        scroll = self.panel_scroll
        self.panel.ensurePolished()
        scroll.setMinimumWidth(max(PANEL_MIN_WIDTH, self.panel.minimumSizeHint().width()
                                   + scroll.verticalScrollBar().sizeHint().width()
                                   + 2 * scroll.frameWidth()))

    def _build_map_group(self):
        group = QGroupBox('1. Model and map')
        layout = QVBoxLayout(group)
        self.model_combo = QComboBox()
        fit_combo(self.model_combo)
        self.model_combo.setToolTip('Trained model folders: env.config, policy.config and weights')
        self.model_browse_button = QPushButton('Browse…')
        self.model_load_button = QPushButton('Load')
        self.model_browse_button.clicked.connect(self._browse_model)
        self.model_load_button.clicked.connect(self.load_selected_model)
        row = QHBoxLayout()
        row.addWidget(self.model_combo, 1)
        row.addWidget(self.model_browse_button)
        row.addWidget(self.model_load_button)
        layout.addLayout(row)
        self.policy_combo = QComboBox()
        for name, label in POLICIES:
            self.policy_combo.addItem(label, name)
        self.policy_combo.setToolTip('SARL + COLREGs filters the actions by the steering '
                                     'rules (Rules 13-17). Switching keeps the scenario.')
        self.policy_combo.activated.connect(self._on_policy_chosen)
        row = QFormLayout()
        row.addRow('Policy', self.policy_combo)
        layout.addLayout(row)
        self.model_label = QLabel('No model loaded')
        self.model_label.setWordWrap(True)
        layout.addWidget(self.model_label)
        self.map_combo = QComboBox()
        fit_combo(self.map_combo)
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
        self.count_spin.setRange(0, 50)
        self.count_spin.setValue(4)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2 ** 31 - 1)
        self.seed_spin.setValue(0)
        form.addRow('Count', self.count_spin)
        form.addRow('seed', self.seed_spin)
        # placement limits: lower spacing / wider spawn band / more tries fit more ships
        self.spacing_spin = QDoubleSpinBox()
        self.spacing_spin.setRange(0.0, 20.0)
        self.spacing_spin.setSingleStep(0.5)
        self.spacing_spin.setValue(3.0)
        self.spacing_spin.setSuffix(' m')
        self.spacing_spin.setToolTip('Minimum distance between ship start/goal points')
        self.cross_min_spin = QDoubleSpinBox()
        self.cross_min_spin.setRange(0.5, MAP_SIZE)
        self.cross_min_spin.setSingleStep(0.5)
        self.cross_min_spin.setValue(6.0)
        self.cross_min_spin.setSuffix(' m')
        self.cross_max_spin = QDoubleSpinBox()
        self.cross_max_spin.setRange(0.5, MAP_SIZE)
        self.cross_max_spin.setSingleStep(0.5)
        self.cross_max_spin.setValue(14.0)
        self.cross_max_spin.setSuffix(' m')
        for spin in (self.cross_min_spin, self.cross_max_spin):
            spin.setToolTip('Distance from the route at which a ship spawns')
        self.trials_spin = QSpinBox()
        self.trials_spin.setRange(1, 100000)
        self.trials_spin.setValue(400)
        self.trials_spin.setToolTip('Random placement attempts per ship before it is skipped')
        form.addRow('spacing', self.spacing_spin)
        form.addRow('spawn min', self.cross_min_spin)
        form.addRow('spawn max', self.cross_max_spin)
        form.addRow('tries / ship', self.trials_spin)
        layout.addLayout(form)
        self.generate_button = QPushButton('Generate random')
        self.generate_button.clicked.connect(self.generate_obstacles)
        layout.addWidget(self.generate_button)
        # a third click mode, exclusive with Set start / Set goal: two clicks per ship
        self.ship_radio = QRadioButton('Place ship by hand')
        self.ship_radio.setToolTip('Click the ship\'s start on the map, then its goal. '
                                   'Generating random ships replaces the ones placed by hand')
        self.point_group.addButton(self.ship_radio)
        self.ship_radio.toggled.connect(self._on_mode_change)
        self.remove_ship_button = QPushButton('Remove last')
        self.remove_ship_button.clicked.connect(self.remove_last_ship)
        layout.addWidget(self.ship_radio)
        self.placed_label = QLabel('Placed 0 / 0')
        row = QHBoxLayout()
        row.addWidget(self.placed_label, 1)
        row.addWidget(self.remove_ship_button)
        layout.addLayout(row)
        self.obstacle_table = QTableWidget(0, 4)
        self.obstacle_table.setHorizontalHeaderLabels(['start x', 'start y', 'goal x', 'goal y'])
        self.obstacle_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.obstacle_table.setMaximumHeight(150)
        # four equal columns that fit the panel, so the table never scrolls sideways
        self.obstacle_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.obstacle_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.obstacle_table.verticalHeader().setDefaultSectionSize(22)
        self.obstacle_table.verticalHeader().setMinimumWidth(24)
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
            label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
            self.result_labels[key] = label
            layout.addRow(title, label)
        return group

    def _build_export_group(self):
        group = QGroupBox('7. Export')
        layout = QVBoxLayout(group)
        self.export_button = QPushButton('Export MP4…')
        self.export_button.clicked.connect(self.export_video)
        layout.addWidget(self.export_button)
        self.export_results_button = QPushButton('Export results…')
        self.export_results_button.setToolTip('CSV: one metrics row, as in the batch results. '
                                              'JSON: settings, metrics, route, ships and the '
                                              'trajectory of every ship.')
        self.export_results_button.clicked.connect(self.export_results)
        layout.addWidget(self.export_results_button)
        return group

    # --------------------------------------------------------------- helpers

    def _log(self, message, level='info'):
        """One timestamped log line; warnings and errors are prefixed and coloured,
        'detail' (tracebacks) is greyed out under the error it explains."""
        text = html.escape(LOG_PREFIXES.get(level, '') + str(message))
        # appendHtml collapses whitespace: keep line breaks and traceback indentation
        text = text.replace('\n', '<br>').replace('  ', '&nbsp;&nbsp;')
        theme = THEMES[self.theme_name]
        if level in theme:
            text = '<span style="color: {};">{}</span>'.format(theme[level], text)
        self.log.appendHtml('<span style="color: {};">{}</span>&nbsp;&nbsp;{}'.format(
            theme['stamp'], time.strftime('%H:%M:%S'), text))

    def _set_status(self, message, error=False, level=None):
        """Show ``message`` in the status bar; warnings and errors also go to the log."""
        level = level or ('error' if error else 'info')
        self.status.setText(message)
        self._status_level = level
        self._color_status()
        if level != 'info':
            self._log(message, level)

    def _color_status(self):
        theme = THEMES[self.theme_name]
        self.status.setStyleSheet('color: {};'.format(theme[self._status_level])
                                  if self._status_level in theme else '')

    # ----------------------------------------------------------------- theme

    def set_theme(self, name):
        """Switch the widgets, the log colours and the map to the 'light' or 'dark' theme.
        Lines already in the log keep the colours they were written in."""
        if name not in THEMES:
            raise ValueError('Unknown theme {!r}, expected one of {}'.format(name, list(THEMES)))
        self.theme_name = name
        theme = THEMES[name]
        app = QApplication.instance()
        app.setPalette(dark_palette() if name == 'dark' else app.style().standardPalette())
        self.panel.setStyleSheet(PRIMARY_BUTTON_STYLE.format(**theme))
        self._fit_panel()
        self._color_status()
        self.view.theme = theme
        self.view.apply_theme()

    def _follow_system_theme(self):
        name = system_theme()
        if name != self.theme_name:
            self.set_theme(name)
            self._log('Switched to the {} theme to match Windows'.format(name))

    def _warn(self, message):
        """Something the user can fix or retry, as opposed to a failure."""
        self._set_status(message, level='warning')

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
        self.model_combo.setEnabled(not running)
        self.model_browse_button.setEnabled(not running)
        self.model_load_button.setEnabled(not running)
        self.policy_combo.setEnabled(not running)
        self.browse_button.setEnabled(not running)
        self.map_combo.setEnabled(not running)
        self.start_radio.setEnabled(not running and state != 'no_map')
        self.goal_radio.setEnabled(not running and state != 'no_map')
        self.plan_button.setEnabled(not running and both_points)
        self.generate_button.setEnabled(not running and has_route)
        if not has_route and self.ship_radio.isChecked():
            self.start_radio.setChecked(True)      # no route, no ships to place
        self.ship_radio.setEnabled(not running and has_route)
        self.remove_ship_button.setEnabled(
            not running and has_route and bool(self.session.scenario['dyn']))
        self.count_spin.setEnabled(not running and has_route)
        self.seed_spin.setEnabled(not running and has_route)
        for spin in (self.spacing_spin, self.cross_min_spin, self.cross_max_spin,
                     self.trials_spin):
            spin.setEnabled(not running and has_route)
        self.run_button.setEnabled(not running and has_route)
        self.cancel_button.setEnabled(running)
        self.steps_spin.setEnabled(not running)
        self.gpu_box.setEnabled(not running)
        self.export_button.setEnabled(not running and self.session.result is not None)
        self.export_results_button.setEnabled(not running and self.session.result is not None)
        playing = self.view.scene is not None
        self.play_button.setEnabled(playing)
        self.frame_slider.setEnabled(playing)
        self.speed_combo.setEnabled(playing)

    def _refresh_after_change(self):
        self._ship_start = None          # the redraw drops a half-placed ship's marker
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
            # the row number in the ship's map colour, so a row is easy to find on the map
            number = QTableWidgetItem(str(i + 1))
            number.setForeground(QColor(PLAY_COLORS[i % len(PLAY_COLORS)]))
            font = number.font()
            font.setBold(True)
            number.setFont(font)
            self.obstacle_table.setVerticalHeaderItem(i, number)
            for j, value in enumerate(row):
                item = QTableWidgetItem('{:.1f}'.format(value))
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.obstacle_table.setItem(i, j, item)
        if rows and self.session.last_seed is None:
            self.placed_label.setText('Placed {} by hand'.format(len(rows)))
        else:
            requested = self.count_spin.value() if self.session.route is not None else 0
            self.placed_label.setText('Placed {} / {}'.format(len(rows), requested))

    def _clear_results(self):
        for label in self.result_labels.values():
            label.setText('—')
        self.result_labels['outcome'].setStyleSheet('')

    def _show_results(self, metrics):
        for key, label in self.result_labels.items():
            label.setText(format_metric(key, metrics.get(key, '')))
        outcome = str(metrics.get('outcome', ''))
        color = ('#2e7d32' if outcome == 'success' else
                 '#c62828' if outcome.startswith('collision') else '#ef6c00')
        self.result_labels['outcome'].setStyleSheet(
            'background: {}; color: white; font-weight: 600; border-radius: 4px;'
            ' padding: 2px 8px;'.format(color))

    # ------------------------------------------------------------------- map

    def _browse_map(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Choose a map PNG', STATIC_MAP_DIR,
                                              'PNG (*.png);;All files (*)')
        if not path:
            return
        if path not in self._map_paths:
            self._map_paths.append(path)
            self.map_combo.addItem(os.path.basename(path))
            set_item_tooltips(self.map_combo, self._map_paths)
        self.map_combo.setCurrentIndex(self._map_paths.index(path))
        self.load_selected_map()

    def load_selected_map(self, *_):
        index = self.map_combo.currentIndex()
        if not (0 <= index < len(self._map_paths)):
            self._warn('No map to load')
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

    # ----------------------------------------------------------------- model

    def _session_like_current(self, model_dir):
        s = self.session
        return ScenarioSession(model_dir, policy_name=s.policy_name,
                               safety_distance=s.safety_distance, colregs=s.colregs,
                               local_goal_dist=s.local_goal_dist, monitor_dist=s.monitor_dist,
                               monitor_exit_dist=s.monitor_exit_dist,
                               lookahead_rate=s.lookahead_rate, shore_clearance=s.shore_clearance,
                               robot_visible=s.robot_visible,
                               device=None if s.device is None else str(s.device))

    @staticmethod
    def _same_path(a, b):
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))

    def _model_index(self, path):
        return next((i for i, p in enumerate(self._model_paths) if self._same_path(p, path)), -1)

    def _add_model_path(self, path):
        if self._model_index(path) < 0:
            self._model_paths.append(path)
            self.model_combo.addItem(os.path.basename(os.path.normpath(path)))
            set_item_tooltips(self.model_combo, self._model_paths)
        return self._model_index(path)

    def _show_model(self):
        self.model_label.setText('Model {} ({})'.format(
            os.path.basename(os.path.normpath(self.session.model_dir)),
            getattr(self.session.policy, 'name', self.session.policy_name)))
        index = self.policy_combo.findData(self.session.policy_name)
        if index < 0:     # a policy started from the command line that the list lacks
            self.policy_combo.addItem(self.session.policy_name, self.session.policy_name)
            index = self.policy_combo.count() - 1
        self.policy_combo.setCurrentIndex(index)

    def _on_policy_chosen(self, index):
        self.load_policy(self.policy_combo.itemData(index))

    def load_policy(self, name):
        """Switch the policy; the map, start, goal, route and ships stay."""
        if self._is_running():
            return False
        if name == self.session.policy_name:
            return True
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            policy = None if self._policy_builder is None else self._policy_builder(name)
            self.session.set_policy(name, policy=policy)
        except ScenarioError as error:
            self._fail(error)
            self._show_model()       # the list goes back to the policy still in use
            return False
        finally:
            QApplication.restoreOverrideCursor()
        self._stop_playback()
        self._show_model()
        self._log('Policy {}'.format(getattr(self.session.policy, 'name', name)))
        if self.session.plan_grid is not None:
            self._refresh_after_change()          # the scenario stays, the old run goes
        else:
            self._sync_widgets()
        self._set_status('Sailing with {} - run again to compare'.format(
            self.policy_combo.currentText()))
        return True

    def _browse_model(self):
        path = QFileDialog.getExistingDirectory(self, 'Choose a trained model folder', MODEL_ROOT)
        if not path:
            return
        problem = model_dir_problem(path)
        if problem:
            self._fail(problem)
            return
        self.model_combo.setCurrentIndex(self._add_model_path(path))
        self.load_selected_model()

    def load_selected_model(self, *_):
        index = self.model_combo.currentIndex()
        if not (0 <= index < len(self._model_paths)):
            self._warn('No model to load')
            return
        self.load_model(self._model_paths[index])

    def load_model(self, path):
        """Switch to the model in ``path``; the map stays, the scenario is cleared."""
        if self._is_running():
            return False
        if self._same_path(path, self.session.model_dir):
            self._set_status('Model {} is already loaded'.format(os.path.basename(path)))
            return True
        problem = model_dir_problem(path)
        if problem:
            self._fail(problem)
            return False
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            session = self._session_factory(path)
        except ScenarioError as error:
            self._fail(error)
            return False
        except Exception as error:      # a folder whose weights do not fit the network
            self._fail('Could not load the model in {}: {}'.format(path, error))
            self._log(traceback.format_exc(), 'detail')
            return False
        finally:
            QApplication.restoreOverrideCursor()
        self._stop_playback()
        self.view.scene = None
        map_path = self.session.map_path
        self.session = session
        self._show_model()
        self._log('Loaded model {}'.format(path))
        if map_path is not None and self._load_map_file(map_path):
            self._set_status('Model {} loaded - set the start and goal again'.format(
                os.path.basename(path)))
        else:
            self._clear_results()
            self._sync_widgets()
        return True

    # ---------------------------------------------------------------- points

    def _on_mode_change(self, *_):
        self.mode = ('start' if self.start_radio.isChecked() else
                     'ship' if self.ship_radio.isChecked() else 'goal')
        if self._ship_start is not None:
            self._ship_start = None
            self.view.draw_map(self.session)      # drop the half-placed ship's marker

    def _on_click(self, event):
        if getattr(event, 'inaxes', None) is not self.ax:
            return
        if getattr(event, 'button', 1) != 1:
            return
        if self._is_running():
            self._warn('Episode running - cancel it or wait for it to finish')
            return
        if self.session.plan_grid is None:
            self._warn('Load a map first')
            return
        world = ScenarioSession.world_from_click(event.xdata, event.ydata)
        if world is None:
            self._warn('That point is off the canvas')
            return
        if self.mode == 'ship':
            self._place_ship_click(world)
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

    def _place_ship_click(self, world):
        """First click: the ship's start, marked on the map. Second click: its goal."""
        if self._ship_start is None:
            if self.session.plan_grid.is_on_land(world):
                self._fail('Ship start ({:.1f}, {:.1f}) is land or inside the safety belt - '
                           'click open water'.format(*world))
                return
            self._ship_start = world
            self.ax.plot(*world, marker='o', markersize=8, markerfacecolor='none',
                         markeredgecolor='#333333', linestyle='None', zorder=6)
            self.canvas.draw_idle()
            self._set_status('Ship start ({:.1f}, {:.1f}) - now click its goal'.format(*world))
            return
        try:
            self.session.add_obstacle(self._ship_start, world)
        except ScenarioError as error:
            self._fail(error)        # the start stays; click another goal
            return
        self._log('Ship placed by hand: ({:.2f}, {:.2f}) -> ({:.2f}, {:.2f})'.format(
            self._ship_start[0], self._ship_start[1], world[0], world[1]))
        self._ship_start = None
        self._refresh_after_change()
        self._set_status('Placed ship {} - click the next ship\'s start'.format(
            len(self.session.scenario['dyn'])))

    def remove_last_ship(self):
        self._ship_start = None
        if self.session.remove_obstacle() is None:
            self._warn('No obstacle ships to remove')
            return
        self._refresh_after_change()
        self._set_status('Removed the last ship - {} left'.format(len(self.session.scenario['dyn'])))

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
            obstacles = self.session.generate_obstacles(
                n, seed, min_dist=self.spacing_spin.value(),
                cross_dist=(self.cross_min_spin.value(), self.cross_max_spin.value()),
                max_trials=self.trials_spin.value())
        except ScenarioError as error:
            self._fail(error)
            return
        self._refresh_after_change()
        if n == 0:
            self._set_status('No obstacle ships - static map only')
        elif len(obstacles) < n:
            self._warn('Placed {} of {} obstacle ships - there was no room for the rest. Lower '
                       'spacing or spawn min, or raise tries / ship'.format(len(obstacles), n))
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
            self._warn('No CUDA available - falling back to CPU')
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
        self._log(message, 'detail')
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
            self._warn('Nothing to export yet')
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
            self._fail(error)
            self._log(traceback.format_exc(), 'detail')
        else:
            self.progress.setValue(100)
            self._set_status('Video written to {}'.format(written))
            self._log('Video written to {}'.format(written))
        finally:
            QApplication.restoreOverrideCursor()
            self._sync_widgets()

    def export_results(self, path=None):
        """Save the last episode's results as CSV or JSON; ``path`` skips the dialog."""
        result = self.session.result
        if result is None:
            self._warn('Nothing to export yet')
            return
        if path is None:
            path, chosen = QFileDialog.getSaveFileName(
                self, 'Export results', 'episode.csv',
                'CSV - one metrics row (*.csv);;JSON - full record with trajectory (*.json)')
            if not path:
                return
            if not os.path.splitext(path)[1]:
                path += '.json' if 'json' in chosen.lower() else '.csv'
        try:
            written = self.session.export_results(result, path)
        except ScenarioError as error:
            self._fail(error)
            return
        self._set_status('Results written to {}'.format(written))
        self._log('Results written to {}'.format(written))

    # --------------------------------------------------------------- closing

    def closeEvent(self, event):
        self._closing = True
        self.timer.stop()
        self._theme_timer.stop()
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
    parser.add_argument('--colregs', default=False, action='store_true')
    parser.add_argument('--safety_distance', type=int, default=1)
    parser.add_argument('--theme', choices=['system'] + list(THEMES), default='system',
                        help='light or dark; system (default) follows the Windows setting')
    args = parser.parse_args(argv)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    apply_style(app)
    model_dir = args.model_dir or DEFAULT_MODEL_DIR
    try:
        session = ScenarioSession(model_dir, safety_distance=args.safety_distance,
                                  colregs=args.colregs,
                                  policy_name='sarl_colregs' if args.colregs else 'sarl',
                                  device='cuda:0' if args.gpu else 'cpu')
    except Exception as error:
        QMessageBox.critical(None, 'CrowdNav', 'Could not load the policy:\n{}\n\n{}'.format(
            error, traceback.format_exc()))
        return 1

    maps = sorted(glob.glob(os.path.join(STATIC_MAP_DIR, '*.png')))
    window = NavigatorApp(session, maps, theme=args.theme)
    if args.map:
        window._map_paths.insert(0, args.map)
        window.map_combo.insertItem(0, os.path.basename(args.map))
        set_item_tooltips(window.map_combo, window._map_paths)
        window.map_combo.setCurrentIndex(0)
    if window._map_paths:
        window.load_selected_map()
    window.resize(1280, 860)
    window.show()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
