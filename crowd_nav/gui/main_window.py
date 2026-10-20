from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QHBoxLayout, QListWidget, QMainWindow, QStackedWidget, QWidget

from crowd_nav.gui.screens.results_screen import ResultsScreen
from crowd_nav.gui.screens.setup_screen import SetupScreen
from crowd_nav.gui.screens.visualize_screen import VisualizeScreen

# Screens shown in the sidebar, in order. Add a screen by appending its class
# here and to the QStackedWidget in _build_ui below.
SCREENS = [SetupScreen, VisualizeScreen, ResultsScreen]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CrowdNav")
        self.resize(1100, 720)
        self._build_ui()

    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.nav_list = QListWidget()
        self.nav_list.setFixedWidth(180)
        self.nav_list.setIconSize(QSize(0, 0))

        self.stack = QStackedWidget()
        for screen_cls in SCREENS:
            self.nav_list.addItem(screen_cls.title)
            self.stack.addWidget(screen_cls())

        self.nav_list.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav_list.setCurrentRow(0)

        root_layout.addWidget(self.nav_list)
        root_layout.addWidget(self.stack, stretch=1)
