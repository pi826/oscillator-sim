"""2D canvas shared by the S1 and metric-graph modes.

Draws the embedded curve (one item per graph edge so edges can be colored
by direction polarization) plus the oscillators as a scatter plot. When
``interactive`` is set (simulation paused), left click emits an add request
and right click a remove request in data coordinates.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor

# direction colors in graph mode: sigma = +1 warm, sigma = -1 cool
SIGMA_POS_COLOR = (255, 140, 0)
SIGMA_NEG_COLOR = (70, 130, 255)

_CURVE_PEN_WIDTH = 2


def phase_brushes(theta: np.ndarray) -> list[QColor]:
    """Color wheel: hue = theta / 2*pi."""
    hues = np.mod(theta, 2.0 * np.pi) / (2.0 * np.pi)
    return [QColor.fromHsvF(float(h), 0.9, 1.0) for h in hues]


def sigma_brushes(sigma: np.ndarray) -> list[QColor]:
    pos = QColor(*SIGMA_POS_COLOR)
    neg = QColor(*SIGMA_NEG_COLOR)
    return [pos if s > 0 else neg for s in sigma]


def polarization_color(p: float) -> QColor:
    """Diverging map for p in [-1, 1]: blue (-1) - gray (0) - red (+1)."""
    p = float(np.clip(p, -1.0, 1.0))
    base = np.array([150.0, 150.0, 150.0])
    warm = np.array([230.0, 60.0, 40.0])
    cool = np.array([50.0, 90.0, 230.0])
    rgb = base + (warm - base) * p if p >= 0 else base + (cool - base) * (-p)
    return QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))


class Canvas2D(pg.PlotWidget):
    addRequested = Signal(object)  # np.ndarray shape (2,)
    removeRequested = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAspectLocked(True)
        self.hideAxis("bottom")
        self.hideAxis("left")
        self.setBackground("k")
        # right click is "remove oscillator", not the pyqtgraph context menu
        self.getPlotItem().setMenuEnabled(False)
        self.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.PanMode)
        self.interactive = False

        self._edge_items: list[pg.PlotDataItem] = []
        self._scatter = pg.ScatterPlotItem(pen=None, size=12)
        self.addItem(self._scatter)
        self.scene().sigMouseClicked.connect(self._on_click)

    def set_curves(self, polylines: list[np.ndarray]) -> None:
        """Replace the drawn curve; one polyline per graph edge (or a single
        one in S1 mode)."""
        for item in self._edge_items:
            self.removeItem(item)
        self._edge_items = []
        for line in polylines:
            item = pg.PlotDataItem(
                line[:, 0], line[:, 1], pen=pg.mkPen((180, 180, 180), width=_CURVE_PEN_WIDTH)
            )
            self.addItem(item)
            item.setZValue(-1)
            self._edge_items.append(item)

    def set_edge_polarizations(self, p: np.ndarray) -> None:
        for item, value in zip(self._edge_items, p):
            item.setPen(pg.mkPen(polarization_color(float(value)), width=_CURVE_PEN_WIDTH + 1))

    def set_points(self, pos: np.ndarray, brushes: list[QColor]) -> None:
        self._scatter.setData(pos=pos, brush=brushes)

    def _on_click(self, ev) -> None:
        if not self.interactive:
            return
        vb = self.getPlotItem().getViewBox()
        p = vb.mapSceneToView(ev.scenePos())
        point = np.array([p.x(), p.y()])
        if ev.button() == Qt.MouseButton.LeftButton:
            ev.accept()
            self.addRequested.emit(point)
        elif ev.button() == Qt.MouseButton.RightButton:
            ev.accept()
            self.removeRequested.emit(point)
