"""Entry point: ``uv run python -m oscillator_sim [options]``.

Without options the full simulator GUI starts. With options the initial
configuration can be chosen from the command line, and a compact
"viewer" mode (canvas only, autoplay) is available for keeping a
simulation running on the desktop, e.g.::

    uv run python -m oscillator_sim --space graph --curve lissajous --viewer

Use ``--install-startup`` with the same options to register the exact
command to run automatically at Windows login.
"""

from __future__ import annotations

import argparse
import os
import sys

STARTUP_SCRIPT_NAME = "oscillator-sim.vbs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m oscillator_sim",
        description="Interactive phase-oscillator simulator (S1 / metric graph / S2).",
        # abbreviations like --install would survive the exact-string filter
        # in install_startup and get baked into the login command
        allow_abbrev=False,
    )
    sim = parser.add_argument_group("initial configuration (names match fuzzily)")
    sim.add_argument("--space", help="state space: circle / graph / sphere")
    sim.add_argument("--model", help="oscillator model (S1) or sphere model")
    sim.add_argument("--curve", help="embedded curve, e.g. 'lissajous'")
    sim.add_argument("--coupling", help="graph coupling function")
    sim.add_argument("--branching", help="graph branching rule")
    sim.add_argument("--omega", help="omega distribution: Identical / Normal")
    sim.add_argument("--rotation", help="sphere rotation mode")
    sim.add_argument("-n", "--n", type=int, help="number of oscillators")
    sim.add_argument("--seed", type=int, help="random seed")
    sim.add_argument("--speed", type=float, help="integration steps per frame (0.1..1000)")
    sim.add_argument("--fps", type=int, help="frame rate of the render timer (default ~30)")
    sim.add_argument("--play", action="store_true", help="start playing immediately")
    sim.add_argument("--paused", action="store_true", help="stay paused (overrides viewer autoplay)")

    win = parser.add_argument_group("window")
    win.add_argument("--viewer", action="store_true",
                     help="canvas-only compact mode (autoplay unless --paused; "
                          "Space pauses, Esc quits)")
    win.add_argument("--frameless", action="store_true", help="no window frame")
    win.add_argument("--on-top", action="store_true", help="keep above all windows")
    win.add_argument("--desktop", action="store_true",
                     help="frameless, bottom-most, no taskbar entry (desktop widget)")
    win.add_argument("--wallpaper", action="store_true",
                     help="fullscreen transparent overlay at the bottom of the z-order: "
                          "only the curve and oscillators are drawn, the real wallpaper "
                          "and desktop icons stay visible, clicks pass through; quit "
                          "from the tray icon")
    win.add_argument("--size", help="window size WxH, e.g. 900x700")
    win.add_argument("--pos", help="window position X,Y, e.g. 60,60")
    win.add_argument("--resolution", type=float, default=None,
                     help="curve sampling multiplier (display smoothness), e.g. 2")

    misc = parser.add_argument_group("misc")
    misc.add_argument("--list", action="store_true", help="list registered names and exit")
    misc.add_argument("--install-startup", action="store_true",
                      help="register this command (minus this flag) to run at Windows login")
    misc.add_argument("--uninstall-startup", action="store_true",
                      help="remove the Windows login registration")
    return parser


# --- startup registration (Windows) ------------------------------------------


def _startup_script_path() -> str:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        raise SystemExit("APPDATA is not set; startup registration needs Windows")
    return os.path.join(
        appdata, "Microsoft", "Windows", "Start Menu", "Programs", "Startup",
        STARTUP_SCRIPT_NAME,
    )


def _quote_cmd_arg(arg: str) -> str:
    return f'"{arg}"' if any(ch in arg for ch in " ()&") else arg


def install_startup(argv: list[str]) -> str:
    """Write a .vbs into the user's Startup folder that reruns this command
    (console-less, via pythonw) at every login."""
    pythonw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
    if not os.path.exists(pythonw):
        pythonw = sys.executable
    kept = [a for a in argv if a not in ("--install-startup", "--uninstall-startup")]
    command = " ".join(
        _quote_cmd_arg(part) for part in [pythonw, "-m", "oscillator_sim", *kept]
    )
    # in VBS string literals, embedded double quotes are doubled
    vbs = (
        'CreateObject("WScript.Shell").Run "'
        + command.replace('"', '""')
        + '", 0, False\r\n'
    )
    path = _startup_script_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # UTF-16 with BOM: wscript.exe reads it correctly even when the venv
    # path contains non-ASCII characters (a UTF-8 .vbs would not be)
    with open(path, "w", encoding="utf-16") as fh:
        fh.write(vbs)
    return path


def uninstall_startup() -> str | None:
    path = _startup_script_path()
    if os.path.exists(path):
        os.remove(path)
        return path
    return None


# --- name resolution -----------------------------------------------------------


def _resolve(kind: str, name: str | None, choices: list[str]) -> str | None:
    """Case-insensitive exact-or-unique-substring match against a registry."""
    if name is None:
        return None
    low = name.lower()
    exact = [c for c in choices if c.lower() == low]
    if exact:
        return exact[0]
    matches = [c for c in choices if low in c.lower()]
    if len(matches) == 1:
        return matches[0]
    raise SystemExit(
        f"--{kind} {name!r} is {'ambiguous' if matches else 'unknown'};"
        f" choices: {', '.join(choices)}"
    )


def main() -> int:
    ns = build_parser().parse_args()

    if ns.uninstall_startup:
        removed = uninstall_startup()
        print(f"removed {removed}" if removed else "no startup registration found")
        return 0

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    # import implementation modules so they register themselves
    from .core import branching, coupling, models, sphere_models  # noqa: F401
    from .core.omega import OMEGA_MODES, SPHERE_ROTATION_MODES
    from .geometry import curves  # noqa: F401
    from .gui.main_window import _SPACE_MODES, LaunchOptions, MainWindow
    from .registry import BRANCHING_RULES, CURVES, GRAPH_COUPLINGS, MODELS, SPHERE_MODELS

    spaces = list(_SPACE_MODES)
    if ns.list:
        for kind, names in (
            ("spaces", spaces),
            ("models (S1)", MODELS.names()),
            ("models (sphere)", SPHERE_MODELS.names()),
            ("curves", CURVES.names()),
            ("couplings", GRAPH_COUPLINGS.names()),
            ("branching rules", BRANCHING_RULES.names()),
            ("omega modes", OMEGA_MODES),
            ("rotation modes", SPHERE_ROTATION_MODES),
        ):
            print(f"{kind}: {', '.join(names)}")
        return 0

    space = _resolve("space", ns.space, spaces)
    mode = _SPACE_MODES[space] if space is not None else "circle"

    # reject configuration flags that would be silently ignored in the
    # selected state space (a wrong-space --model would otherwise no-op
    # and a *different* simulation would run than the one asked for)
    for flag, value, allowed in (
        ("model", ns.model, ("circle", "sphere")),
        ("curve", ns.curve, ("circle", "graph")),
        ("omega", ns.omega, ("circle", "graph")),
        ("rotation", ns.rotation, ("sphere",)),
        ("coupling", ns.coupling, ("graph",)),
        ("branching", ns.branching, ("graph",)),
    ):
        if value is not None and mode not in allowed:
            raise SystemExit(f"--{flag} does not apply to the '{mode}' state space")

    model_choices = MODELS.names() if mode == "circle" else SPHERE_MODELS.names()
    launch = LaunchOptions(
        space=space,
        model=_resolve("model", ns.model, model_choices),
        curve=_resolve("curve", ns.curve, CURVES.names()),
        coupling=_resolve("coupling", ns.coupling, GRAPH_COUPLINGS.names()),
        branching=_resolve("branching", ns.branching, BRANCHING_RULES.names()),
        omega_mode=_resolve("omega", ns.omega, OMEGA_MODES),
        rotation_mode=_resolve("rotation", ns.rotation, SPHERE_ROTATION_MODES),
        n=ns.n,
        seed=ns.seed,
        speed=ns.speed,
        fps=ns.fps,
        resolution=ns.resolution,
        play=(ns.play or ns.viewer or ns.desktop or ns.wallpaper) and not ns.paused,
        viewer=ns.viewer or ns.desktop or ns.wallpaper,
        transparent=ns.wallpaper,
    )

    # register for login startup only after the names above validated, so a
    # typo cannot get baked into a command that fails invisibly at login
    if ns.install_startup:
        path = install_startup(sys.argv[1:])
        print(f"registered for login startup: {path}")
        return 0

    app = QApplication(sys.argv)
    window = MainWindow(launch)

    flags = Qt.WindowType.Window
    if ns.frameless or ns.desktop or ns.wallpaper:
        flags |= Qt.WindowType.FramelessWindowHint
    if ns.on_top:
        flags |= Qt.WindowType.WindowStaysOnTopHint
    if ns.desktop or ns.wallpaper:
        flags |= Qt.WindowType.WindowStaysOnBottomHint | Qt.WindowType.Tool
    if ns.wallpaper:
        # never take focus and let every click fall through to the desktop
        # icons below
        flags |= (
            Qt.WindowType.WindowDoesNotAcceptFocus
            | Qt.WindowType.WindowTransparentForInput
        )
    if flags != Qt.WindowType.Window:
        window.setWindowFlags(flags)

    if ns.wallpaper:
        window.setGeometry(app.primaryScreen().geometry())
    else:
        window.resize(1280, 820)
        if ns.viewer or ns.desktop:
            window.resize(720, 560)
        if ns.size:
            try:
                w, h = (int(v) for v in ns.size.lower().split("x"))
                window.resize(w, h)
            except ValueError:
                raise SystemExit(f"--size {ns.size!r}: expected WxH, e.g. 900x700")
        if ns.pos:
            try:
                x, y = (int(v) for v in ns.pos.split(","))
                window.move(x, y)
            except ValueError:
                raise SystemExit(f"--pos {ns.pos!r}: expected X,Y, e.g. 60,60")

    window.show()

    if ns.wallpaper:
        from .gui.wallpaper import pin_to_bottom

        pin_to_bottom(int(window.winId()))

    # a single Ctrl+C quits: the default handler only raises inside a Python
    # callback (our frame timer), where the KeyboardInterrupt gets swallowed
    import signal

    signal.signal(signal.SIGINT, lambda *_: app.quit())

    tray = None
    if launch.viewer:
        from .gui.tray import create_tray

        tray = create_tray(app, window)  # noqa: F841 - keep the reference alive

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
