from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QLabel, QComboBox, QFileDialog, QHBoxLayout, QPushButton, QVBoxLayout

from crowd_nav.gui,map_loader import BUNDLED_MAPS_DIR, list_bundled_maps, load_static_map
from crowd_nav.gui.screens.base_screen import BaseScreen
from crowd_nav.gui.widgets.map_canvas import MapCanvas

BROWSE_LABEL = "Browse..."


class SetupScreen(BaseScreen):
    """Placeholder for choosing policy, running parameters, and loading static maps.
    
    Map loading is functional: it resuses the same HybridEnv.load_from_image
    path test.py/evaluate.py use, so the land grid shown here is exactly what
    the simulator would plan and collide against for the chosen map
    """

    title = "Setup / Run"

    map_loaded = pyqtSignal(object) # emits the loaded hybridEnv

    def __init__(self, parent=None):
        self.current_map = None
        super().__init__(parent)

    def build_content(self):
        policy_placeholder = QLabel(
            "Policy and run-parameter controls (mirrors train.py / test.py CLI "
            "options) go here."
        )
        placeholder.setWordWrap(True)
        self.content_layout.addWidget(placeholder)

        policy_placeholder.setWordWrap(True)
        self.content_layout.addWidget(policy_placeholder)

        map_label = QLabel("Static map")
        map_label.setStyleSheet("font-weight: 600; margin-top: 12px;")
        self.content_layout.addWidget(map_label)

        picker_row = QHBoxLayout()
        self.map_combo = QComboBox()
        self._bundled_paths = list_bundled_maps()
        for map_path in self._bundled_paths:
            self.map_combo.addItem(map_path.name)
        self.map_combo.addItem(BROWSE_LABEL)
        picker_row.addWidget(self.map_combo, stretch=1)

        self.load_button = QPushButton("Load Map")
        self.load_button.clicked.connect(self._on_load_clicked)
        picker_row.addWidget(self.load_button)
        self.content_layout.addLayout(picker_row)

        self.status_label = QLabel("No map loaded.")
        self.status_label.setWordWrap(True)
        self.content_layout.addWidget(self.status_label)

        self.preview_canvas = MapCanvas()
        self.preview_canvas.setFixedHeight(260)
        self.content_layout.addWidget(self.preview_canvas)

        if self._bundled_paths:
            self._load_map(self._bundled_paths[0])

    def _on_load_clicked(self):
        if self.map_combo.currentText() == BROWSE_LABEL:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Choose static map", str(BUNDLED_MAPS_DIR), "PNG images (*.png)"
            )
            if not file_path:
                return
            self._load_map(file_path)
            return

        index = self.map_combo.currentIndex()
        if 0 <= index < len(self._bundled_paths):
            self._load_map(self._bundled_paths[index])

    def _load_map(self, path):
        try:
            env = load_static_map(path)
        except (FileNotFoundError, OSError) as exc:
            self.status_label.setText(f"Failed to load {path}: {exc}")
            return

        land_cells = len(env.obstacles)
        self.status_label.setText(
            f"Loaded {path} — {env.x_range}x{env.y_range} grid @ {env.resolution} m/cell "
            f"({env.x_max:g} m x {env.y_max:g} m), {land_cells} land cells."
        )
        self.preview_canvas.set_map(env)
        self.current_map = env
        self.map_loaded.emit(env)
