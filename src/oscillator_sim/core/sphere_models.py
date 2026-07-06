"""Swarm-sphere (Lohe) dynamics on S2.

States are unit vectors x_i (rows of an (n, 3) array). The intrinsic
rotation Omega_i in so(3) is stored as its axial vector w_i, acting as
Omega_i x = w_i x x (cross product). After every RK4 step the states are
renormalized so the sphere constraint |x_i| = 1 holds exactly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from ..registry import SPHERE_MODELS
from .integrator import RK4Stepper, Stepper
from .params import Parameterized, ParamSpec
from .simulation import Dynamics


class SphereModel(Parameterized, ABC):
    """Vector field dx_i/dt on (S2)^n; the coupling term is what varies."""

    name: ClassVar[str]

    def __init__(self, rotations: np.ndarray) -> None:
        super().__init__()
        self.rotations = np.asarray(rotations, dtype=np.float64)  # (n, 3)

    @abstractmethod
    def dxdt(self, x: np.ndarray, t: float) -> np.ndarray: ...


@SPHERE_MODELS.register
class LoheModel(SphereModel):
    """dx_i/dt = Omega_i x_i + (K/n) sum_j [x_j - <x_j, x_i> x_i]."""

    name = "Lohe"
    params = {"K": ParamSpec("K (coupling)", 1.0, -10.0, 10.0, 0.1)}

    def dxdt(self, x: np.ndarray, t: float) -> np.ndarray:
        # (K/n) sum_j [x_j - <x_j, x_i> x_i] = K [m - <m, x_i> x_i], m = mean
        m = x.mean(axis=0)
        coupling = self.values["K"] * (m[None, :] - (x @ m)[:, None] * x)
        return np.cross(self.rotations, x) + coupling


@SPHERE_MODELS.register
class LohePhaseLagModel(SphereModel):
    """Lohe coupling with a phase-lag rotation inserted:
    dx_i/dt = Omega_i x_i + (K/n) sum_j [R x_j - <R x_j, x_i> x_i],
    where R = R_z(alpha) rotates about the z-axis by the lag angle alpha.
    alpha = 0 reduces to the plain Lohe model; alpha != 0 makes the
    population swirl around the z-axis while ordering (the S2 analogue of
    the Sakaguchi phase lag).
    """

    name = "Lohe (phase lag)"
    params = {
        "K": ParamSpec("K (coupling)", 1.0, -10.0, 10.0, 0.1),
        "alpha": ParamSpec("alpha (lag angle)", 0.5, -3.2, 3.2, 0.05),
    }

    def dxdt(self, x: np.ndarray, t: float) -> np.ndarray:
        m = x.mean(axis=0)
        a = self.values["alpha"]
        c, s = np.cos(a), np.sin(a)
        m_rot = np.array([c * m[0] - s * m[1], s * m[0] + c * m[1], m[2]])
        coupling = self.values["K"] * (m_rot[None, :] - (x @ m_rot)[:, None] * x)
        return np.cross(self.rotations, x) + coupling


class SphereDynamics(Dynamics):
    """RK4 step followed by the mandatory renormalization onto S2."""

    def __init__(self, model: SphereModel, stepper: Stepper | None = None) -> None:
        self.model = model
        self.stepper: Stepper = stepper if stepper is not None else RK4Stepper()

    def step(self, state: np.ndarray, t: float, dt: float) -> np.ndarray:
        x = self.stepper.step(self.model.dxdt, state, t, dt)
        return x / np.linalg.norm(x, axis=1, keepdims=True)
