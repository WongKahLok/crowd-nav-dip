from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class BaseScreen(QWidget):
    """Common shell for a navigable screen: a title label plus a content area.

    Subclasses set `title` and build their real content in `build_content`.
    """

    title = "Screen"

    # Set False in subclasses whose content should expand to fill the screen
    # (e.g. a canvas) rather than sit at its natural size with empty space below.
    trailing_stretch = True

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        heading = QLabel(self.title)
        heading.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(heading)

        self.content_layout = QVBoxLayout()
        layout.addLayout(self.content_layout)
        layout.addStretch(1)

        self.build_content()

    def build_content(self):
        """Override in subclasses to populate self.content_layout."""
        pass
