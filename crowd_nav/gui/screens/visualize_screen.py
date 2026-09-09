from PyQt6.QtWidgets import QLabel

from crowd_nav.gui.screens.base_screen import BaseScreen
from crowd_nav.gui.widgets.map_canvas import MapCanvas


class VisualizeScreen(BaseScreen):
    """Placeholder for live/replayed simulation visualization.
    
    Shows the static map loaded on the Setup screen as a backdrop; ships and
    trajectories are not wired in yet
    """

    title = "Visualize"
    trailing_stretch = False

    def build_content(self):
        placeholder = QLabel(
            "Simulation canvas (live run or replay of a test_case) goes here."
            "The land map below updates when you load one on the Setup screen."
        )
        placeholder.setWordWrap(True)
        self.content_layout.addWidget(placeholder)

        self.canvas = MapCanvas()
        self.content_layout.addWidget(self.canvas, stretch=1)

    def set_map(self, env):
        self.canvas.set_map(env)
