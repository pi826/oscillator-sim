"""Keep a window glued to the bottom of the z-order (wallpaper mode).

Historical note: the classic trick of re-parenting a window into the
desktop's WorkerW layer (behind the icons) no longer works on recent
Windows 11 builds - the window renders into its surface but DWM never
composites children injected into Progman/WorkerW, so nothing shows up
on screen (verified on build 26200). Additionally, while embedded that
way, QApplication.quit() is silently ignored and the process cannot
exit its event loop. Wallpaper mode therefore uses a full-screen,
frameless, non-activating window pinned to the bottom of the z-order:
it covers the desktop icons but reliably shows below every other window.
"""

from __future__ import annotations

import ctypes
import sys

_HWND_BOTTOM = 1
_SWP_NOSIZE = 0x0001
_SWP_NOMOVE = 0x0002
_SWP_NOACTIVATE = 0x0010


def pin_to_bottom(win_id: int) -> bool:
    """Push the window to the bottom of the z-order (Windows)."""
    if sys.platform != "win32":
        return False
    try:
        user32 = ctypes.windll.user32
        user32.SetWindowPos.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        return bool(
            user32.SetWindowPos(
                ctypes.c_void_p(win_id),
                ctypes.c_void_p(_HWND_BOTTOM),
                0,
                0,
                0,
                0,
                _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOACTIVATE,
            )
        )
    except Exception:
        return False
