"""Best-effort embedding of a window behind the desktop icons (Windows).

Uses the classic wallpaper-engine trick: message 0x052C makes Progman
spawn a WorkerW layer behind the icon list (SHELLDLL_DefView); our window
is then re-parented into that layer. On newer Windows 11 builds the
DefView can live directly under Progman, in which case Progman itself is
used as the parent. Returns False on any failure so callers can fall back
to a normal bottom-most window.
"""

from __future__ import annotations

import ctypes
import sys


def embed_into_desktop(win_id: int) -> bool:
    if sys.platform != "win32":
        return False
    try:
        user32 = ctypes.windll.user32
        user32.FindWindowW.restype = ctypes.c_void_p
        user32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        user32.FindWindowExW.restype = ctypes.c_void_p
        user32.FindWindowExW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
        ]
        user32.SetParent.restype = ctypes.c_void_p
        user32.SetParent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        progman = user32.FindWindowW("Progman", None)
        if not progman:
            return False

        # ask Progman to create the WorkerW layer behind the desktop icons
        result = ctypes.c_ulonglong()
        user32.SendMessageTimeoutW(
            ctypes.c_void_p(progman), 0x052C, 0, 0, 0, 1000, ctypes.byref(result)
        )

        found: list[int] = []
        proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)

        def _enum(hwnd: int, _lparam: int) -> int:
            defview = user32.FindWindowExW(hwnd, None, "SHELLDLL_DefView", None)
            if defview:
                worker = user32.FindWindowExW(None, hwnd, "WorkerW", None)
                if worker:
                    found.append(worker)
            return 1

        user32.EnumWindows(proto(_enum), 0)
        target = found[0] if found else progman
        return bool(user32.SetParent(ctypes.c_void_p(win_id), ctypes.c_void_p(target)))
    except Exception:
        return False
