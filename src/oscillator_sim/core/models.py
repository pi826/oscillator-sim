"""Coupled oscillator models on S1.

Subclass ``OscillatorModel``, declare ``name`` and ``params``, implement
``dtheta`` and register with ``@MODELS.register``; the model then appears
in the GUI dropdown automatically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from ..constants import COT_CLIP
from ..registry import MODELS
from .params import Parameterized, ParamSpec

TWO_PI = 2.0 * np.pi


class OscillatorModel(Parameterized, ABC):
    """dtheta_i/dt for n phase oscillators with natural frequencies omega_i.

    Second-order models (``second_order = True``) are integrated on the
    extended state (theta, v) by ``CircleDynamics``; they implement
    ``accel`` instead of a meaningful ``dtheta``.
    """

    name: ClassVar[str]
    second_order: ClassVar[bool] = False

    def __init__(self, omega: np.ndarray) -> None:
        super().__init__()
        self.omega = np.asarray(omega, dtype=np.float64)

    @abstractmethod
    def dtheta(self, theta: np.ndarray, t: float) -> np.ndarray: ...

    def accel(self, theta: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
        raise NotImplementedError("only second-order models define accel()")


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


@MODELS.register
class CotangentNearestNeighborModel(OscillatorModel):
    """dtheta_i/dt = (1/2)[cot((theta_i - theta_{i-1})/2) - cot((theta_{i+1} - theta_i)/2)].

    Indices are cyclic (theta_{n+1} = theta_0, theta_{-1} = theta_n). The
    equation has no omega term, so the omega distribution is ignored.
    cot(x/2) is 2*pi-periodic in x, hence the phase differences may be
    wrapped freely; the interaction repels neighbors on both sides and the
    flow relaxes toward the splay state (equal gaps). The exact field
    diverges when neighbors collide, so |cot| is clipped to COT_CLIP for
    fixed-step stability.
    """

    name = "Cotangent (nearest neighbor)"
    params: ClassVar[dict[str, ParamSpec]] = {}

    def dtheta(self, theta: np.ndarray, t: float) -> np.ndarray:
        if theta.size < 2:
            return np.zeros_like(theta)
        gap_prev = np.mod(theta - np.roll(theta, 1), TWO_PI)  # theta_i - theta_{i-1}
        gap_next = np.mod(np.roll(theta, -1) - theta, TWO_PI)  # theta_{i+1} - theta_i
        with np.errstate(divide="ignore"):
            cot_prev = np.clip(1.0 / np.tan(gap_prev / 2.0), -COT_CLIP, COT_CLIP)
            cot_next = np.clip(1.0 / np.tan(gap_next / 2.0), -COT_CLIP, COT_CLIP)
        return 0.5 * (cot_prev - cot_next)


@MODELS.register
class KuramotoDaidoModel(OscillatorModel):
    """Two-harmonic Daido coupling:
    dtheta_i/dt = omega_i + (1/n) sum_j [K1 sin(d_ji) + K2 sin(2 d_ji)],
    d_ji = theta_j - theta_i. A dominant second harmonic locks phases mod
    pi and produces two-cluster ((m,2)-pattern) states.
    """

    name = "Kuramoto-Daido (2 harmonics)"
    params = {
        "K1": ParamSpec("K1 (1st harmonic)", 0.5, -10.0, 10.0, 0.1),
        "K2": ParamSpec("K2 (2nd harmonic)", 1.0, -10.0, 10.0, 0.1),
    }

    def dtheta(self, theta: np.ndarray, t: float) -> np.ndarray:
        n = theta.size
        z1 = np.exp(1j * theta).sum()
        z2 = np.exp(2j * theta).sum()
        h1 = np.imag(np.exp(-1j * theta) * z1)
        h2 = np.imag(np.exp(-2j * theta) * z2)
        return self.omega + (self.values["K1"] / n) * h1 + (self.values["K2"] / n) * h2


@MODELS.register
class WinfreeModel(OscillatorModel):
    """Winfree model with the classic pulse/response pair:
    dtheta_i/dt = omega_i + (K/n) sum_j (1 + cos theta_j) * (-sin theta_i).
    """

    name = "Winfree"
    params = {"K": ParamSpec("K (coupling)", 0.5, -10.0, 10.0, 0.05)}

    def dtheta(self, theta: np.ndarray, t: float) -> np.ndarray:
        n = theta.size
        pulse = (1.0 + np.cos(theta)).sum()
        return self.omega + (self.values["K"] / n) * pulse * (-np.sin(theta))


@MODELS.register
class ActiveRotatorModel(OscillatorModel):
    """Active rotators (Shinomoto-Kuramoto):
    dtheta_i/dt = omega_i - a sin(theta_i) + (K/n) sum_j sin(theta_j - theta_i).
    For a > omega_i the units are excitable rather than oscillatory.
    """

    name = "Active rotator"
    params = {
        "K": ParamSpec("K (coupling)", 1.0, -10.0, 10.0, 0.1),
        "a": ParamSpec("a (pinning)", 0.8, -10.0, 10.0, 0.05),
    }

    def dtheta(self, theta: np.ndarray, t: float) -> np.ndarray:
        n = theta.size
        z = np.exp(1j * theta).sum()
        coupling = np.imag(np.exp(-1j * theta) * z)
        return self.omega - self.values["a"] * np.sin(theta) + (self.values["K"] / n) * coupling


@MODELS.register
class RingKuramotoModel(OscillatorModel):
    """Kuramoto coupling restricted to index neighbors on a ring:
    dtheta_i/dt = omega_i + (K/2)[sin(theta_{i+1} - theta_i) + sin(theta_{i-1} - theta_i)].
    Supports twisted states theta_j = 2*pi*m*j/n ((m,n)-patterns).
    """

    name = "Kuramoto (nearest neighbor)"
    params = {"K": ParamSpec("K (coupling)", 1.0, -10.0, 10.0, 0.1)}

    def dtheta(self, theta: np.ndarray, t: float) -> np.ndarray:
        if theta.size < 2:
            return self.omega.copy()
        nxt = np.roll(theta, -1)
        prev = np.roll(theta, 1)
        return self.omega + (self.values["K"] / 2.0) * (np.sin(nxt - theta) + np.sin(prev - theta))


@MODELS.register
class NonlocalRingModel(OscillatorModel):
    """Kuramoto-Battogtokh nonlocal coupling on the index ring:
    dtheta_i/dt = omega_i - K sum_j G_ij sin(theta_i - theta_j + alpha),
    G_ij proportional to exp(-kappa * d_ij) with the ring distance
    d_ij = min(|i-j|, n-|i-j|)/n and rows normalized to sum 1.

    Near alpha = pi/2 with moderate kappa this hosts chimera states:
    a synchronized domain coexisting with an incoherent one.
    """

    name = "Nonlocal ring (chimera)"
    params = {
        "K": ParamSpec("K (coupling)", 1.0, -10.0, 10.0, 0.1),
        "alpha": ParamSpec("alpha (phase lag)", 1.45, -3.2, 3.2, 0.01),
        "kappa": ParamSpec("kappa (inverse range)", 4.0, 0.0, 50.0, 0.5),
    }

    def __init__(self, omega: np.ndarray) -> None:
        super().__init__(omega)
        self._kernel: np.ndarray | None = None
        self._kernel_key: tuple[int, float] | None = None

    def _kernel_for(self, n: int) -> np.ndarray:
        key = (n, self.values["kappa"])
        if self._kernel_key != key:
            idx = np.arange(n)
            d = np.abs(idx[:, None] - idx[None, :])
            d = np.minimum(d, n - d) / n
            g = np.exp(-self.values["kappa"] * d)
            g /= g.sum(axis=1, keepdims=True)
            self._kernel = g
            self._kernel_key = key
        return self._kernel  # type: ignore[return-value]

    def dtheta(self, theta: np.ndarray, t: float) -> np.ndarray:
        g = self._kernel_for(theta.size)
        # sum_j G_ij sin(theta_i - theta_j + alpha)
        #   = Im(exp(i(theta_i + alpha)) * sum_j G_ij exp(-i theta_j))
        z = g @ np.exp(-1j * theta)
        lagged = np.exp(1j * (theta + self.values["alpha"]))
        return self.omega - self.values["K"] * np.imag(lagged * z)


@MODELS.register
class InertialKuramotoModel(OscillatorModel):
    """Kuramoto with inertia (power-grid / swing equation form):
    m theta''_i + theta'_i = omega_i + (K/n) sum_j sin(theta_j - theta_i).

    Second order: CircleDynamics integrates the pair (theta, v) and keeps
    the velocities internally (they start at v_i = omega_i, i.e. free
    rotation, and restart there when the oscillator count changes).
    """

    name = "Kuramoto (inertia)"
    second_order = True
    params = {
        "K": ParamSpec("K (coupling)", 2.0, -10.0, 10.0, 0.1),
        "m": ParamSpec("m (inertia)", 1.0, 0.01, 20.0, 0.05),
    }

    def dtheta(self, theta: np.ndarray, t: float) -> np.ndarray:
        raise NotImplementedError("second-order model; integrated via accel()")

    def accel(self, theta: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
        n = theta.size
        z = np.exp(1j * theta).sum()
        coupling = (self.values["K"] / n) * np.imag(np.exp(-1j * theta) * z)
        return (self.omega + coupling - v) / self.values["m"]
