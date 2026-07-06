"""3D canvas for the sphere mode (pyqtgraph.opengl GLViewWidget).

A translucent sphere mesh with the oscillators as a point cloud; left drag
orbits the camera. When ``interactive`` is set (paused), a left click casts
a ray from the camera and adds an oscillator at the near-side intersection
with the sphere; a right click removes the on-screen nearest oscillator.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QMatrix4x4, QVector3D

_CLICK_SLOP_PX = 6.0


def direction_colors(x: np.ndarray) -> np.ndarray:
    """Per-point RGBA from position on the sphere (soft direction coding)."""
    rgb = 0.25 + 0.75 * (x + 1.0) / 2.0
    return np.concatenate([rgb, np.ones((x.shape[0], 1))], axis=1)


class Canvas3D(gl.GLViewWidget):
    addRequested = Signal(object)  # np.ndarray shape (3,)
    removeIndexRequested = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.interactive = False
        self.setCameraPosition(distance=4.0)

        mesh = gl.MeshData.sphere(rows=24, cols=48)
        self._sphere = gl.GLMeshItem(
            meshdata=mesh,
            smooth=True,
            color=(0.35, 0.55, 1.0, 0.25),
            shader="shaded",
            glOptions="translucent",
        )
        self.addItem(self._sphere)

        self._points = np.zeros((0, 3))
        self._scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)), size=9.0, pxMode=True, glOptions="opaque"
        )
        self.addItem(self._scatter)
        self._press_pos: QPointF | None = None

    def set_points(self, pos: np.ndarray, colors: np.ndarray) -> None:
        self._points = pos
        if pos.shape[0] == 0:
            self._scatter.setData(pos=np.zeros((1, 3)), color=np.zeros((1, 4)))
        else:
            self._scatter.setData(pos=pos, color=colors)

    # --- picking -------------------------------------------------------------

    def mousePressEvent(self, ev) -> None:
        self._press_pos = ev.position()
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev) -> None:
        pressed = self._press_pos
        self._press_pos = None
        if (
            self.interactive
            and pressed is not None
            and (ev.position() - pressed).manhattanLength() < _CLICK_SLOP_PX
        ):
            if ev.button() == Qt.MouseButton.LeftButton:
                point = self._pick_sphere(ev.position())
                if point is not None:
                    self.addRequested.emit(point)
            elif ev.button() == Qt.MouseButton.RightButton:
                index = self._nearest_screen_index(ev.position())
                if index is not None:
                    self.removeIndexRequested.emit(index)
        super().mouseReleaseEvent(ev)

    def _mvp(self) -> QMatrix4x4:
        viewport = self.getViewport()
        return self.projectionMatrix(viewport, viewport) * self.viewMatrix()

    def _pick_sphere(self, pos: QPointF) -> np.ndarray | None:
        """Near-side intersection of the click ray with the unit sphere."""
        w = max(self.width(), 1)
        h = max(self.height(), 1)
        ndc_x = 2.0 * pos.x() / w - 1.0
        ndc_y = 1.0 - 2.0 * pos.y() / h
        inv, ok = self._mvp().inverted()
        if not ok:
            return None
        near = inv.map(QVector3D(ndc_x, ndc_y, -1.0))
        far = inv.map(QVector3D(ndc_x, ndc_y, 1.0))
        origin = np.array([near.x(), near.y(), near.z()])
        direction = np.array([far.x(), far.y(), far.z()]) - origin
        direction /= np.linalg.norm(direction)

        b = float(origin @ direction)
        c = float(origin @ origin) - 1.0
        disc = b * b - c
        if disc < 0.0:
            return None
        t = -b - np.sqrt(disc)  # smaller root = near side
        if t < 0.0:
            t = -b + np.sqrt(disc)
            if t < 0.0:
                return None
        point = origin + t * direction
        return point / np.linalg.norm(point)

    def _nearest_screen_index(self, pos: QPointF) -> int | None:
        if self._points.shape[0] == 0:
            return None
        mvp = self._mvp()
        w = max(self.width(), 1)
        h = max(self.height(), 1)
        screen = np.empty((self._points.shape[0], 2))
        for i, p in enumerate(self._points):
            v = mvp.map(QVector3D(float(p[0]), float(p[1]), float(p[2])))
            screen[i] = ((v.x() + 1.0) / 2.0 * w, (1.0 - v.y()) / 2.0 * h)
        d = np.linalg.norm(screen - np.array([pos.x(), pos.y()]), axis=1)
        return int(np.argmin(d))
