"""Fixed-step ODE steppers.

``Stepper`` is the seam for a future native (e.g. Rust / pyo3) backend:
anything that maps ``(f, y, t, dt) -> y_next`` on NumPy arrays can be
substituted without touching the rest of the code.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

Derivative = Callable[[np.ndarray, float], np.ndarray]


class Stepper(Protocol):
    def step(self, f: Derivative, y: np.ndarray, t: float, dt: float) -> np.ndarray: ...


class RK4Stepper:
    """Classic fourth-order Runge-Kutta with fixed step size."""

    def step(self, f: Derivative, y: np.ndarray, t: float, dt: float) -> np.ndarray:
        k1 = f(y, t)
        k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(y + dt * k3, t + dt)
        return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
