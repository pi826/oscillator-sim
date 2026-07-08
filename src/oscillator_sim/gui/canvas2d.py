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

pg.setConfigOptions(antialias=True)

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


def loop_colors(n_loops: int) -> list[QColor]:
    """Palette used for loops/arcs: warm/cool for two, a hue wheel otherwise."""
    if n_loops <= 2:
        return [QColor(*SIGMA_POS_COLOR), QColor(*SIGMA_NEG_COLOR)][:n_loops]
    return [QColor.fromHsvF(k / n_loops, 0.9, 1.0) for k in range(n_loops)]


def loop_brushes(loop: np.ndarray, n_loops: int) -> list[QColor]:
    """One color per loop/arc index."""
    palette = loop_colors(n_loops)
    return [palette[int(k) % n_loops] for k in loop]


def _arrow_pose(line: np.ndarray) -> tuple[np.ndarray, float]:
    """Midpoint (by arclength) of a polyline and the ArrowItem angle that
    points along the direction of travel there.

    ArrowItem's angle-0 arrow points left and rotations are applied in
    Qt's y-down item space, so a data-space tangent (dx, dy) needs
    angle = 180 - atan2(dy, dx) in degrees.
    """
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    i = int(np.clip(np.searchsorted(cum, cum[-1] / 2.0), 1, len(line) - 1))
    d = line[i] - line[i - 1]
    if np.linalg.norm(d) == 0.0:
        d = np.array([1.0, 0.0])
    angle = 180.0 - float(np.degrees(np.arctan2(d[1], d[0])))
    return line[i], angle


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
        # right click is "remove oscillator", not the pyqtgraph context menu;
        # the corner "A" button would re-enable auto-range (view wobble)
        self.getPlotItem().setMenuEnabled(False)
        self.getPlotItem().hideButtons()
        self.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.PanMode)
        self.interactive = False

        self._edge_items: list[pg.PlotDataItem] = []
        self._arrow_items: list[pg.ArrowItem] = []
        self._pol_cache: np.ndarray | None = None
        self._scatter = pg.ScatterPlotItem(pen=None, size=12)
        self.addItem(self._scatter)
        self.scene().sigMouseClicked.connect(self._on_click)

    def set_transparent(self) -> None:
        """Fully transparent background (wallpaper overlay mode): only the
        curve and the oscillators are drawn. QGraphicsView needs the whole
        recipe - brushless background alone still erases the viewport with
        an opaque palette color."""
        self.setBackground(None)
        self.setStyleSheet("background: transparent")
        for widget in (self, self.viewport()):
            widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            widget.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
            widget.setAutoFillBackground(False)

    def set_curves(
        self,
        polylines: list[np.ndarray],
        show_arrows: bool = False,
        arrow_colors: list[QColor] | None = None,
    ) -> None:
        """Replace the drawn curve; one polyline per graph edge (or a single
        one in S1 mode). With ``show_arrows``, one arrowhead per polyline
        marks the arc's orientation (direction of increasing s / alpha) at
        its arclength midpoint."""
        for item in self._edge_items + self._arrow_items:
            self.removeItem(item)
        self._edge_items = []
        self._arrow_items = []
        self._pol_cache = None
        for line in polylines:
            item = pg.PlotDataItem(
                line[:, 0], line[:, 1], pen=pg.mkPen((180, 180, 180), width=_CURVE_PEN_WIDTH)
            )
            self.addItem(item)
            item.setZValue(-1)
            self._edge_items.append(item)
        if show_arrows:
            for k, line in enumerate(polylines):
                pos, angle = _arrow_pose(line)
                color = arrow_colors[k] if arrow_colors else QColor(235, 235, 235)
                arrow = pg.ArrowItem(
                    angle=angle,
                    headLen=15,
                    tipAngle=32,
                    brush=color,
                    pen=pg.mkPen(QColor(0, 0, 0, 140), width=1),
                    pxMode=True,
                )
                arrow.setPos(float(pos[0]), float(pos[1]))
                arrow.setZValue(-0.5)
                self.addItem(arrow)
                self._arrow_items.append(arrow)
        self._fix_view_range(polylines)

    def _fix_view_range(self, polylines: list[np.ndarray]) -> None:
        """Freeze the view on the curve's bounding box.

        With auto-range left on, every scatter/pen update re-runs the range
        computation and the view rectangle drifts a little each frame (most
        visibly when oscillators cross a vertex and edge pens change), which
        reads as the whole graph wobbling. Manual pan/zoom still works.
        """
        points = np.vstack(polylines)
        (x0, y0), (x1, y1) = points.min(axis=0), points.max(axis=0)
        vb = self.getPlotItem().getViewBox()
        vb.disableAutoRange()
        vb.setRange(xRange=(float(x0), float(x1)), yRange=(float(y0), float(y1)), padding=0.08)

    def set_edge_polarizations(self, p: np.ndarray) -> None:
        # quantize and update only the edges whose value actually moved, so
        # steady frames do not trigger repaints / bound recomputations
        quantized = np.round(np.clip(np.asarray(p, dtype=np.float64), -1.0, 1.0) * 32.0) / 32.0
        cache = self._pol_cache
        for k, (item, value) in enumerate(zip(self._edge_items, quantized)):
            if cache is not None and k < cache.size and cache[k] == value:
                continue
            item.setPen(pg.mkPen(polarization_color(float(value)), width=_CURVE_PEN_WIDTH + 1))
        self._pol_cache = quantized

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
