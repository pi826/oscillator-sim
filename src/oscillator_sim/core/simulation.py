"""GUI-independent simulation facade.

``Dynamics`` is the "advance a state array by one step" interface shared by
all state spaces; a native (Rust / pyo3) implementation can replace any of
its subclasses without touching the GUI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .integrator import RK4Stepper, Stepper
from .models import OscillatorModel

TWO_PI = 2.0 * np.pi


class Dynamics(ABC):
    """Advances a state (opaque to the GUI) by one time step."""

    @abstractmethod
    def step(self, state: Any, t: float, dt: float) -> Any: ...


class CircleDynamics(Dynamics):
    """S1 dynamics: RK4 on dtheta, phases kept in [0, 2*pi).

    Second-order models are integrated on the extended state (theta, v);
    the velocities are kept here (initialized to omega = free rotation,
    and re-initialized whenever the oscillator count changes).
    """

    def __init__(self, model: OscillatorModel, stepper: Stepper | None = None) -> None:
        self.model = model
        self.stepper: Stepper = stepper if stepper is not None else RK4Stepper()
        self._velocity: np.ndarray | None = None

    def step(self, state: np.ndarray, t: float, dt: float) -> np.ndarray:
        if self.model.second_order:
            return self._step_second_order(state, t, dt)
        theta = self.stepper.step(self.model.dtheta, state, t, dt)
        return np.mod(theta, TWO_PI)

    def _step_second_order(self, state: np.ndarray, t: float, dt: float) -> np.ndarray:
        n = state.size
        if self._velocity is None or self._velocity.size != n:
            self._velocity = self.model.omega.copy()

        def f(y: np.ndarray, t_: float) -> np.ndarray:
            theta, v = y[:n], y[n:]
            return np.concatenate([v, self.model.accel(theta, v, t_)])

        y = self.stepper.step(f, np.concatenate([state, self._velocity]), t, dt)
        self._velocity = y[n:]
        return np.mod(y[:n], TWO_PI)


class Simulation:
    """Bundles dynamics, state, time and the seeded RNG; GUI calls step()."""

    def __init__(
        self,
        dynamics: Dynamics,
        state: Any,
        dt: float,
        rng: np.random.Generator,
    ) -> None:
        self.dynamics = dynamics
        self.state = state
        self.dt = dt
        self.rng = rng
        self.t = 0.0

    def step(self, n_steps: int = 1) -> None:
        for _ in range(n_steps):
            self.state = self.dynamics.step(self.state, self.t, self.dt)
            self.t += self.dt
