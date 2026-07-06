"""Main window: orchestrates state space, dynamics, canvases and panels.

The QTimer drives rendering at a fixed frame rate; the number of
integration steps per frame (speed slider) is independent of it.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..constants import DEFAULT_DT
from ..core.graph_dynamics import GraphDynamics
from ..core.observables import (
    classify_circle,
    classify_sphere,
    edge_polarization,
    ks_statistic,
    order_parameter,
    sphere_order_parameter,
    uniformity,
)
from ..core.omega import make_omega, make_rotations
from ..core.simulation import CircleDynamics, Simulation
from ..core.sphere_models import SphereDynamics
from ..registry import BRANCHING_RULES, CURVES, GRAPH_COUPLINGS, MODELS, SPHERE_MODELS
from ..space.circle import Circle
from ..space.graph import MetricGraph
from ..space.sphere import Sphere
from .canvas2d import Canvas2D, phase_brushes, sigma_brushes
from .canvas3d import Canvas3D, direction_colors
from .controls import ControlPanel
from .status_panel import StatusPanel

FRAME_INTERVAL_MS = 33
MAX_STEPS_PER_FRAME = 2000
CLASSIFY_EVERY_N_FRAMES = 10

_SPACE_MODES: dict[str, str] = {
    Circle.name: "circle",
    MetricGraph.name: "graph",
    Sphere.name: "sphere",
}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Phase Oscillator Simulator")

        self.canvas2d = Canvas2D()
        self.canvas3d = Canvas3D()
        self.stacked = QStackedWidget()
        self.stacked.addWidget(self.canvas2d)
        self.stacked.addWidget(self.canvas3d)

        self.controls = ControlPanel(spaces=list(_SPACE_MODES))
        self.status = StatusPanel()
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.addWidget(self.controls)
        side_layout.addWidget(self.status)
        side.setFixedWidth(380)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(self.stacked, stretch=1)
        layout.addWidget(side)
        self.setCentralWidget(central)

        self.controls.configChanged.connect(self._rebuild)
        self.controls.resetRequested.connect(self._rebuild)
        self.controls.paramChanged.connect(self._on_param)
        self.controls.playToggled.connect(self._on_play)
        self.controls.placementRequested.connect(self._on_place)
        self.canvas2d.addRequested.connect(self._on_add)
        self.canvas2d.removeRequested.connect(self._on_remove)
        self.canvas3d.addRequested.connect(self._on_add)
        self.canvas3d.removeIndexRequested.connect(self._on_remove_index)

        self._accum = 0.0
        self._frame = 0
        self._timer = QTimer(self)
        self._timer.setInterval(FRAME_INTERVAL_MS)
        self._timer.timeout.connect(self._on_tick)

        self._rebuild()
        self._timer.start()

    # --- simulation construction -----------------------------------------

    def _rebuild(self) -> None:
        mode = _SPACE_MODES[self.controls.config().space]
        self.mode = mode
        self.controls.set_mode(mode)
        cfg = self.controls.config()
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)

        if mode == "circle":
            curve = CURVES.get(cfg.curve)()
            self.space = Circle(curve)
            omega = make_omega(cfg.omega_mode, cfg.n, rng)
            self.model = MODELS.get(cfg.model)(omega)
            dynamics = CircleDynamics(self.model)
            state = self.space.initial_states(cfg.n, rng, "random")
            self.canvas2d.set_curves([self.space.curve_polyline()])
            self.status.set_series_name("r1")
            self.stacked.setCurrentWidget(self.canvas2d)
        elif mode == "graph":
            curve = CURVES.get(cfg.curve)()
            self.space = MetricGraph(curve, rng)
            omega = make_omega(cfg.omega_mode, cfg.n, rng)
            coupling = GRAPH_COUPLINGS.get(cfg.coupling)()
            branching = BRANCHING_RULES.get(cfg.branching)()
            self.model = GraphDynamics(self.space, omega, coupling, branching, rng)
            dynamics = self.model
            state = self.space.initial_states(cfg.n, rng, "random")
            self.canvas2d.set_curves([e.points for e in self.space.edges])
            self.status.set_series_name("U")
            self.stacked.setCurrentWidget(self.canvas2d)
        elif mode == "sphere":
            self.space = Sphere()
            rotations = make_rotations(cfg.rotation_mode, cfg.n, rng)
            self.model = SPHERE_MODELS.get(cfg.model)(rotations)
            dynamics = SphereDynamics(self.model)
            state = self.space.initial_states(cfg.n, rng, "random")
            self.status.set_series_name("r")
            self.stacked.setCurrentWidget(self.canvas3d)
        else:
            raise ValueError(f"unknown mode {mode!r}")

        self.sim = Simulation(dynamics, state, DEFAULT_DT, rng)
        self.controls.build_params(self.model.params, self.model.values)
        self.controls.set_time(0.0)
        self._accum = 0.0
        self._refresh(stepped=True)

    # --- control callbacks ---------------------------------------------------

    def _on_param(self, name: str, value: float) -> None:
        self.model.set_param(name, value)

    def _on_play(self, playing: bool) -> None:
        self.canvas2d.interactive = not playing
        self.canvas3d.interactive = not playing

    def _on_place(self, placement: str) -> None:
        if placement not in self.space.placement_modes:
            return
        n = self.space.count(self.sim.state)
        self.sim.state = self.space.initial_states(n, self.sim.rng, placement)
        self._refresh(stepped=True)

    def _on_add(self, point: np.ndarray) -> None:
        self.sim.state = self.space.add_at(self.sim.state, point, self.sim.rng)
        if self.mode == "sphere":
            extra = make_rotations(self.cfg.rotation_mode, 1, self.sim.rng)
            self.model.rotations = np.vstack([self.model.rotations, extra])
        else:
            extra = make_omega(self.cfg.omega_mode, 1, self.sim.rng)
            self.model.omega = np.append(self.model.omega, extra)
        self._refresh(stepped=True)

    def _on_remove(self, point: np.ndarray) -> None:
        index = self.space.nearest_index(self.sim.state, point)
        self._on_remove_index(index)

    def _on_remove_index(self, index: int) -> None:
        if self.space.count(self.sim.state) <= 1:
            return
        self.sim.state = self.space.remove_index(self.sim.state, index)
        if self.mode == "sphere":
            self.model.rotations = np.delete(self.model.rotations, index, axis=0)
        else:
            self.model.omega = np.delete(self.model.omega, index)
        self._refresh(stepped=True)

    # --- frame loop --------------------------------------------------------

    def _on_tick(self) -> None:
        stepped = False
        if self.controls.playing:
            self._accum += self.controls.speed
            n_steps = min(int(self._accum), MAX_STEPS_PER_FRAME)
            self._accum -= n_steps
            if n_steps > 0:
                self.sim.step(n_steps)
                stepped = True
        self._frame += 1
        self._refresh(stepped=stepped)

    def _refresh(self, stepped: bool) -> None:
        state = self.sim.state
        self.controls.set_time(self.sim.t)
        if self.mode == "circle":
            self.canvas2d.set_points(self.space.positions(state), phase_brushes(state))
            if stepped:
                self.status.append(self.sim.t, order_parameter(state))
                self.status.redraw()
            if self._frame % CLASSIFY_EVERY_N_FRAMES == 0 or not stepped:
                self.status.set_classification(classify_circle(state))
            self.status.set_extra("")
        elif self.mode == "graph":
            self.canvas2d.set_points(self.space.positions(state), sigma_brushes(state.sigma))
            pol = edge_polarization(state.edge, state.sigma, len(self.space.edges))
            self.canvas2d.set_edge_polarizations(pol)
            u_value = uniformity(state.edge, self.space.lengths, self.space.total_length)
            if stepped:
                self.status.append(self.sim.t, u_value)
                self.status.redraw()
            self.status.set_classification(f"U = {u_value:.3f}")
            if self._frame % CLASSIFY_EVERY_N_FRAMES == 0 or not stepped:
                ks = ks_statistic(
                    self.space.sample_pair_distances(state), self.space.reference_distances
                )
                pol_text = ", ".join(f"e{k}: {p:+.2f}" for k, p in enumerate(pol))
                self.status.set_extra(f"KS = {ks:.3f}\npolarization: {pol_text}")
        elif self.mode == "sphere":
            self.canvas3d.set_points(state, direction_colors(state))
            if stepped:
                self.status.append(self.sim.t, sphere_order_parameter(state))
                self.status.redraw()
            if self._frame % CLASSIFY_EVERY_N_FRAMES == 0 or not stepped:
                self.status.set_classification(classify_sphere(state))
            self.status.set_extra("")
