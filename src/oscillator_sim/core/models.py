"""Coupled oscillator models on S1.

Subclass ``OscillatorModel``, declare ``name`` and ``params``, implement
``dtheta`` and register with ``@MODELS.register``; the model then appears
in the GUI dropdown automatically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from ..registry import MODELS
from .params import Parameterized, ParamSpec


class OscillatorModel(Parameterized, ABC):
    """dtheta_i/dt for n phase oscillators with natural frequencies omega_i."""

    name: ClassVar[str]

    def __init__(self, omega: np.ndarray) -> None:
        super().__init__()
        self.omega = np.asarray(omega, dtype=np.float64)

    @abstractmethod
    def dtheta(self, theta: np.ndarray, t: float) -> np.ndarray: ...


@MODELS.register
class KuramotoModel(OscillatorModel):
    """dtheta_i/dt = omega_i + (K/n) sum_j sin(theta_j - theta_i)."""

    name = "Kuramoto"
    params = {"K": ParamSpec("K (coupling)", 1.0, -10.0, 10.0, 0.1)}

    def dtheta(self, theta: np.ndarray, t: float) -> np.ndarray:
        n = theta.size
        # sum_j sin(theta_j - theta_i) = Im(exp(-i theta_i) * sum_j exp(i theta_j))
        z = np.exp(1j * theta).sum()
        return self.omega + (self.values["K"] / n) * np.imag(np.exp(-1j * theta) * z)


@MODELS.register
class SakaguchiKuramotoModel(OscillatorModel):
    """dtheta_i/dt = omega_i + (K/n) sum_j sin(theta_j - theta_i - alpha)."""

    name = "Sakaguchi-Kuramoto"
    params = {
        "K": ParamSpec("K (coupling)", 1.0, -10.0, 10.0, 0.1),
        "alpha": ParamSpec("alpha (phase lag)", 0.5, -3.2, 3.2, 0.05),
    }

    def dtheta(self, theta: np.ndarray, t: float) -> np.ndarray:
        n = theta.size
        z = np.exp(1j * theta).sum()
        phase = np.exp(-1j * (theta + self.values["alpha"]))
        return self.omega + (self.values["K"] / n) * np.imag(phase * z)
