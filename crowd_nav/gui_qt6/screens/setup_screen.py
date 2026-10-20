from PyQt6.QtWidgets import QLabel

from crowd_nav.gui_qt6.screens.base_screen import BaseScreen


class SetupScreen(BaseScreen):
    """Placeholder for choosing policy, static map, and run parameters."""

    title = "Setup / Run"

    def build_content(self):
        placeholder = QLabel(
            "Policy, static map, and parameter controls (mirrors train.py / "
            "test.py CLI options) go here."
        )
        placeholder.setWordWrap(True)
        self.content_layout.addWidget(placeholder)
