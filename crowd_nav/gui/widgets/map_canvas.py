from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QWidget

WATER_COLOR = QColor("#bcd8e8")
LAND_COLOR = QColor("#5a4632")
GRID_LINE_COLOR = QColor(0, 0, 0, 25)


class MapCanvas(QWidget):
    """Renders the land grid a HybridEnv loaded from a static map PNG.

    Draws the same thresholded land cells the simulator plans and collides
    against (not the raw source image), scaled to fit the widget while
    preserving the map's aspect ratio.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._env = None
        self.setMinimumSize(200, 200)

    def set_map(self, env):
        """`env` is a HybridEnv from map_loader.load_static_map, or None to clear."""
        self._env = env
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.rect(), self.palette().window())

        if self._env is None:
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No map loaded")
            return

        env = self._env
        cols, rows = env.x_range, env.y_range

        available_w, available_h = self.width(), self.height()
        cell = max(1.0, min(available_w / cols, available_h / rows))
        draw_w, draw_h = cell * cols, cell * rows
        origin_x = (available_w - draw_w) / 2
        origin_y = (available_h - draw_h) / 2

        painter.fillRect(QRectF(origin_x, origin_y, draw_w, draw_h), WATER_COLOR)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(LAND_COLOR)
        for gx, gy in env.obstacles:
            # flip y so the map matches the simulator's y-up world coordinates
            row_from_top = rows - 1 - gy
            painter.drawRect(QRectF(
                origin_x + gx * cell,
                origin_y + row_from_top * cell,
                cell + 0.5,
                cell + 0.5,
            ))

        painter.setPen(GRID_LINE_COLOR)
        painter.drawRect(QRectF(origin_x, origin_y, draw_w, draw_h))
