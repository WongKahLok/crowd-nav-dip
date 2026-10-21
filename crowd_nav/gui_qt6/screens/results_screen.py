from PyQt6.QtWidgets import QLabel

from crowd_nav.gui_qt6.screens.base_screen import BaseScreen


class ResultsScreen(BaseScreen):
    """Placeholder for evaluate.py-style metrics and comparisons."""

    title = "Results / Metrics"

    def build_content(self):
        placeholder = QLabel(
            "Metrics tables/plots from evaluate.py CSV output go here."
        )
        placeholder.setWordWrap(True)
        self.content_layout.addWidget(placeholder)
