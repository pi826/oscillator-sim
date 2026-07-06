"""Entry point: ``uv run python -m oscillator_sim``."""

from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    # import implementation modules so they register themselves
    from .core import branching, coupling, models  # noqa: F401
    from .geometry import curves  # noqa: F401
    from .gui.main_window import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1280, 820)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
