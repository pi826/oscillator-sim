"""System tray icon for the viewer / desktop / wallpaper modes.

Frameless bottom-most windows have no close button and may never receive
keyboard focus, so the tray menu is the always-available way to pause or
quit a long-running instance.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon


def _lissajous_icon() -> QIcon:
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    t = np.linspace(0.0, 2.0 * np.pi, 256)
    xs = 32.0 + 26.0 * np.sin(3.0 * t)
    ys = 32.0 + 26.0 * np.sin(2.0 * t)
    polygon = QPolygonF([QPointF(float(x), float(y)) for x, y in zip(xs, ys)])
    painter.setPen(QPen(QColor(255, 150, 40), 5.0))
    painter.drawPolyline(polygon)
    painter.end()
    return QIcon(pixmap)


def create_tray(app: QApplication, window) -> QSystemTrayIcon | None:
    """Show a tray icon with pause/quit controls; returns None if the
    platform has no tray. The caller must keep the returned reference."""
    if not QSystemTrayIcon.isSystemTrayAvailable():
        return None
    tray = QSystemTrayIcon(_lissajous_icon(), app)
    tray.setToolTip("Phase Oscillator Simulator")

    menu = QMenu()
    toggle_action = menu.addAction("Pause")
    menu.addSeparator()
    quit_action = menu.addAction("Quit")

    def _sync_text(_playing: bool | None = None) -> None:
        toggle_action.setText("Pause" if window.controls.playing else "Play")

    def _toggle() -> None:
        window.controls.set_playing(not window.controls.playing)

    toggle_action.triggered.connect(_toggle)
    window.controls.playToggled.connect(_sync_text)
    _sync_text()
    quit_action.triggered.connect(app.quit)

    tray.setContextMenu(menu)
    tray.activated.connect(
        lambda reason: _toggle() if reason == QSystemTrayIcon.ActivationReason.Trigger else None
    )
    tray.show()
    tray.showMessage(
        "Phase Oscillator Simulator",
        "Running. Right-click this tray icon to pause or quit.",
        QSystemTrayIcon.MessageIcon.Information,
        4000,
    )
    tray._menu = menu  # keep the menu alive alongside the tray icon
    return tray
