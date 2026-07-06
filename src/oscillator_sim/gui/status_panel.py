"""Status panel: classification label, extra statistics, and a small
time-series window (r_1 for S1, U for graphs, r for the sphere)."""

from __future__ import annotations

from collections import deque

import pyqtgraph as pg
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

HISTORY_LENGTH = 4096


class StatusPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.class_label = QLabel("-")
        self.class_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        layout.addWidget(QLabel("State classification:"))
        layout.addWidget(self.class_label)

        self.extra_label = QLabel("")
        self.extra_label.setWordWrap(True)
        layout.addWidget(self.extra_label)

        self.plot = pg.PlotWidget()
        self.plot.setFixedHeight(160)
        self.plot.setYRange(0.0, 1.05)
        self.plot.setLabel("bottom", "t")
        self._series_name = "r1"
        self.plot.setLabel("left", self._series_name)
        self._curve = self.plot.plot(pen=pg.mkPen((80, 200, 120), width=2))
        layout.addWidget(self.plot)

        self._ts: deque[float] = deque(maxlen=HISTORY_LENGTH)
        self._vs: deque[float] = deque(maxlen=HISTORY_LENGTH)

    def set_series_name(self, name: str) -> None:
        self._series_name = name
        self.plot.setLabel("left", name)
        self.clear_history()

    def clear_history(self) -> None:
        self._ts.clear()
        self._vs.clear()
        self._curve.setData([], [])

    def append(self, t: float, value: float) -> None:
        self._ts.append(t)
        self._vs.append(value)

    def redraw(self) -> None:
        self._curve.setData(list(self._ts), list(self._vs))

    def set_classification(self, text: str) -> None:
        self.class_label.setText(text)

    def set_extra(self, text: str) -> None:
        self.extra_label.setText(text)
        self.extra_label.setVisible(bool(text))
