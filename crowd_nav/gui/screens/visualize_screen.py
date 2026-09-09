from PyQt6.QtWidgets import QLabel

from crowd_nav.gui.screens.base_screen import BaseScreen


class VisualizeScreen(BaseScreen):
    """Placeholder for live/replayed simulation visualization."""

    title = "Visualize"

    def build_content(self):
        placeholder = QLabel(
            "Simulation canvas (live run or replay of a test_case) goes here."
        )
        placeholder.setWordWrap(True)
        self.content_layout.addWidget(placeholder)
