"""Control panel: dropdowns, speed slider, transport and placement buttons.

Model parameters are generated automatically from ``ParamSpec`` declarations,
and dropdown contents come from the registries, so new models / curves /
rules need no changes here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..constants import DEFAULT_N, DEFAULT_RESOLUTION, DEFAULT_SEED, RESOLUTION_CHOICES
from ..core.omega import OMEGA_MODES, SPHERE_ROTATION_MODES
from ..core.params import ParamSpec
from ..registry import (
    BRANCHING_RULES,
    CURVES,
    GLUED_MODELS,
    GRAPH_COUPLINGS,
    MODELS,
    SPHERE_MODELS,
)


@dataclass(frozen=True)
class SimConfig:
    n: int
    space: str
    model: str
    curve: str
    omega_mode: str
    rotation_mode: str
    coupling: str
    branching: str
    seed: int
    resolution: float


class ControlPanel(QWidget):
    configChanged = Signal()
    paramChanged = Signal(str, float)
    curveParamChanged = Signal(str, float)
    playToggled = Signal(bool)
    resetRequested = Signal()
    placementRequested = Signal(str)

    def __init__(self, spaces: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._updating = False

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self._form = form
        layout.addLayout(form)

        self.n_spin = QSpinBox(minimum=1, maximum=2000, value=DEFAULT_N)
        self.space_combo = QComboBox()
        self.space_combo.addItems(spaces)
        self.model_combo = QComboBox()
        self.curve_combo = QComboBox()
        self.curve_combo.addItems(CURVES.names())
        self.omega_combo = QComboBox()
        self.omega_combo.addItems(OMEGA_MODES)
        self.rotation_combo = QComboBox()
        self.rotation_combo.addItems(SPHERE_ROTATION_MODES)
        self.coupling_combo = QComboBox()
        self.coupling_combo.addItems(GRAPH_COUPLINGS.names())
        self.branching_combo = QComboBox()
        self.branching_combo.addItems(BRANCHING_RULES.names())
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems([f"{r:g}x" for r in RESOLUTION_CHOICES])
        self.resolution_combo.setCurrentText(f"{DEFAULT_RESOLUTION:g}x")
        self.seed_spin = QSpinBox(minimum=0, maximum=2**31 - 1, value=DEFAULT_SEED)

        form.addRow("n (oscillators)", self.n_spin)
        form.addRow("State space", self.space_combo)
        form.addRow("Model", self.model_combo)
        form.addRow("Curve", self.curve_combo)
        form.addRow("Curve resolution", self.resolution_combo)
        form.addRow("omega dist.", self.omega_combo)
        form.addRow("Rotations", self.rotation_combo)
        form.addRow("Coupling", self.coupling_combo)
        form.addRow("Branching", self.branching_combo)
        form.addRow("Seed", self.seed_spin)

        for combo in (
            self.space_combo,
            self.model_combo,
            self.curve_combo,
            self.resolution_combo,
            self.omega_combo,
            self.rotation_combo,
            self.coupling_combo,
            self.branching_combo,
        ):
            combo.currentIndexChanged.connect(self._emit_config_changed)
        self.n_spin.valueChanged.connect(self._emit_config_changed)
        self.seed_spin.valueChanged.connect(self._emit_config_changed)

        # --- model / curve parameters (auto-generated spin boxes) --------
        self.param_group = QGroupBox("Model parameters")
        self._param_form = QFormLayout(self.param_group)
        layout.addWidget(self.param_group)

        self.curve_param_group = QGroupBox("Curve parameters")
        self._curve_param_form = QFormLayout(self.curve_param_group)
        layout.addWidget(self.curve_param_group)
        self.curve_param_group.setVisible(False)

        # --- speed (log slider: steps per frame) -------------------------
        self.speed_slider = QSlider(Qt.Orientation.Horizontal, minimum=0, maximum=80, value=40)
        self.speed_label = QLabel()
        self.speed_slider.valueChanged.connect(self._update_speed_label)
        self._update_speed_label()
        speed_row = QHBoxLayout()
        speed_row.addWidget(self.speed_slider)
        speed_row.addWidget(self.speed_label)
        form.addRow("Speed", speed_row)

        # --- transport ----------------------------------------------------
        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self._on_play_toggled)
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.resetRequested)
        self.time_label = QLabel("t = 0.000")
        transport = QHBoxLayout()
        transport.addWidget(self.play_button)
        transport.addWidget(self.reset_button)
        transport.addWidget(self.time_label)
        layout.addLayout(transport)

        # --- placement -----------------------------------------------------
        place_group = QGroupBox("Placement (paused only: click adds, right-click removes)")
        place_row = QHBoxLayout(place_group)
        self.place_buttons: dict[str, QPushButton] = {}
        for mode, text in (("uniform", "Uniform"), ("random", "Random"), ("splay", "Splay")):
            btn = QPushButton(text)
            btn.clicked.connect(lambda _=False, m=mode: self.placementRequested.emit(m))
            place_row.addWidget(btn)
            self.place_buttons[mode] = btn
        layout.addWidget(place_group)
        layout.addStretch(1)

    # --- signal plumbing ---------------------------------------------------

    def _emit_config_changed(self) -> None:
        if not self._updating:
            self.configChanged.emit()

    def _on_play_toggled(self, checked: bool) -> None:
        self.play_button.setText("Pause" if checked else "Play")
        self.playToggled.emit(checked)

    def _update_speed_label(self) -> None:
        self.speed_label.setText(f"{self.speed:.2f} steps/frame")

    # --- public API ----------------------------------------------------------

    @property
    def speed(self) -> float:
        """Integration steps per rendered frame, log scale 0.1 .. 1000."""
        return float(10.0 ** (self.speed_slider.value() / 20.0 - 1.0))

    @property
    def playing(self) -> bool:
        return self.play_button.isChecked()

    def set_playing(self, playing: bool) -> None:
        self.play_button.setChecked(playing)

    def set_time(self, t: float) -> None:
        self.time_label.setText(f"t = {t:.3f}")

    def set_choices(
        self,
        *,
        space: str | None = None,
        model: str | None = None,
        curve: str | None = None,
        omega_mode: str | None = None,
        rotation_mode: str | None = None,
        coupling: str | None = None,
        branching: str | None = None,
        n: int | None = None,
        seed: int | None = None,
        resolution: float | None = None,
    ) -> None:
        """Set several controls at once without emitting configChanged
        (the caller is expected to rebuild once afterwards)."""
        self._updating = True
        try:
            if space is not None:
                self.space_combo.setCurrentText(space)
            if model is not None:
                self.model_combo.setCurrentText(model)
            if curve is not None:
                self.curve_combo.setCurrentText(curve)
            if omega_mode is not None:
                self.omega_combo.setCurrentText(omega_mode)
            if rotation_mode is not None:
                self.rotation_combo.setCurrentText(rotation_mode)
            if coupling is not None:
                self.coupling_combo.setCurrentText(coupling)
            if branching is not None:
                self.branching_combo.setCurrentText(branching)
            if n is not None:
                self.n_spin.setValue(n)
            if seed is not None:
                self.seed_spin.setValue(seed)
            if resolution is not None:
                nearest = min(RESOLUTION_CHOICES, key=lambda r: abs(r - resolution))
                self.resolution_combo.setCurrentText(f"{nearest:g}x")
        finally:
            self._updating = False

    def set_speed(self, steps_per_frame: float) -> None:
        """Move the log slider to the position closest to the given speed."""
        steps_per_frame = min(max(steps_per_frame, 0.1), 1000.0)
        self.speed_slider.setValue(round((math.log10(steps_per_frame) + 1.0) * 20.0))

    def config(self) -> SimConfig:
        return SimConfig(
            n=self.n_spin.value(),
            space=self.space_combo.currentText(),
            model=self.model_combo.currentText(),
            curve=self.curve_combo.currentText(),
            omega_mode=self.omega_combo.currentText(),
            rotation_mode=self.rotation_combo.currentText(),
            coupling=self.coupling_combo.currentText(),
            branching=self.branching_combo.currentText(),
            seed=self.seed_spin.value(),
            resolution=float(self.resolution_combo.currentText().rstrip("x")),
        )

    def set_mode(self, mode: str) -> None:
        """Show only the controls that apply to 'circle' / 'graph' / 'sphere'."""
        self._updating = True
        try:
            model_names = {
                "circle": MODELS.names(),
                "graph": [],
                "sphere": SPHERE_MODELS.names(),
                "glued": GLUED_MODELS.names(),
            }[mode]
            current = self.model_combo.currentText()
            self.model_combo.clear()
            self.model_combo.addItems(model_names)
            if current in model_names:
                self.model_combo.setCurrentText(current)

            form = self._form
            form.setRowVisible(self.model_combo, bool(model_names))
            form.setRowVisible(self.curve_combo, mode in ("circle", "graph", "glued"))
            form.setRowVisible(self.resolution_combo, mode in ("circle", "graph", "glued"))
            form.setRowVisible(self.omega_combo, mode in ("circle", "graph", "glued"))
            form.setRowVisible(self.rotation_combo, mode == "sphere")
            form.setRowVisible(self.coupling_combo, mode == "graph")
            form.setRowVisible(self.branching_combo, mode == "graph")

            self.place_buttons["uniform"].setVisible(mode != "sphere")
            self.place_buttons["splay"].setVisible(mode == "circle")
        finally:
            self._updating = False

    @staticmethod
    def _fill_param_form(
        form: QFormLayout,
        params: dict[str, ParamSpec],
        values: dict[str, float],
        signal: Signal,
    ) -> None:
        while form.rowCount():
            form.removeRow(0)
        for name, spec in params.items():
            box = QDoubleSpinBox(
                minimum=spec.minimum,
                maximum=spec.maximum,
                singleStep=spec.step,
                decimals=spec.decimals,
            )
            box.setValue(values[name])
            box.valueChanged.connect(lambda v, key=name: signal.emit(key, float(v)))
            form.addRow(spec.label, box)

    def build_params(self, params: dict[str, ParamSpec], values: dict[str, float]) -> None:
        """Regenerate one spin box per declared model parameter."""
        self._fill_param_form(self._param_form, params, values, self.paramChanged)
        self.param_group.setVisible(bool(params))

    def build_curve_params(self, params: dict[str, ParamSpec], values: dict[str, float]) -> None:
        """Regenerate one spin box per declared curve parameter (changing
        one rebuilds the simulation, since the geometry changes)."""
        self._fill_param_form(self._curve_param_form, params, values, self.curveParamChanged)
        self.curve_param_group.setVisible(bool(params))
