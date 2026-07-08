"""Dynamics on the glued limacon loops (limacon_branching_kuramoto memo).

State: (loop_i, alpha_i). The local phase alpha flows deterministically
under a mean-field coupling that distinguishes same-loop pairs (K_same)
from cross-loop pairs (K_cross); whenever an oscillator passes the glue
point alpha = 0 (in either direction), its loop is redrawn uniformly at
random - the memo's stochastic boundary condition. The result is a
piecewise-deterministic Markov process, not a plain ODE.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from ..registry import GLUED_MODELS
from ..space.glued import GluedState
from .integrator import RK4Stepper, Stepper
from .params import Parameterized, ParamSpec
from .simulation import Dynamics

TWO_PI = 2.0 * np.pi


class GluedLoopModel(Parameterized, ABC):
    """d alpha_i/dt for oscillators on the glued loops."""

    name: ClassVar[str]

    def __init__(self, omega: np.ndarray) -> None:
        super().__init__()
        self.omega = np.asarray(omega, dtype=np.float64)

    def _k_matrix(self, loop: np.ndarray) -> np.ndarray:
        same = loop[:, None] == loop[None, :]
        return np.where(same, self.values["K_same"], self.values["K_cross"])

    @abstractmethod
    def dalpha(self, alpha: np.ndarray, loop: np.ndarray, t: float) -> np.ndarray: ...


@GLUED_MODELS.register
class GluedKuramoto(GluedLoopModel):
    """d alpha_i/dt = w*omega_i + (1/n) sum_j K(l_i, l_j) sin(alpha_j - alpha_i).

    On each loop this is the memo's sin(2*delta) Kuramoto written in the
    local phase alpha = 2*theta; K_same / K_cross split the interaction
    within a loop from the interaction across the two loops.
    """

    name = "Kuramoto (glued loops)"
    params = {
        "K_same": ParamSpec("K_same (within loop)", 1.0, -10.0, 10.0, 0.1),
        "K_cross": ParamSpec("K_cross (across loops)", 0.5, -10.0, 10.0, 0.1),
        "w": ParamSpec("w (omega scale)", 1.0, 0.0, 5.0, 0.1),
    }

    def dalpha(self, alpha: np.ndarray, loop: np.ndarray, t: float) -> np.ndarray:
        n = alpha.size
        diff = alpha[None, :] - alpha[:, None]  # alpha_j - alpha_i
        return self.values["w"] * self.omega + (self._k_matrix(loop) * np.sin(diff)).sum(
            axis=1
        ) / n


@GLUED_MODELS.register
class GluedCotangent(GluedLoopModel):
    """d alpha_i/dt = w*omega_i + (1/n) sum_j K(l_i, l_j) cot_eps((alpha_i - alpha_j)/2).

    The memo's cotangent flow on the local phase; cot is regularized as
    cot_eps(x) = sin(x) cos(x) / (sin(x)^2 + eps^2), which is smooth,
    bounded by 1/(2*eps) and vanishes at collisions, so the fixed-step
    RK4 stays stable (the raw field diverges there).
    """

    name = "Cotangent (glued loops)"
    params = {
        "K_same": ParamSpec("K_same (within loop)", 1.0, -10.0, 10.0, 0.1),
        "K_cross": ParamSpec("K_cross (across loops)", 0.0, -10.0, 10.0, 0.1),
        "eps": ParamSpec("eps (regularization)", 0.05, 0.001, 1.0, 0.005),
        "w": ParamSpec("w (omega scale)", 0.0, 0.0, 5.0, 0.1),
    }

    def dalpha(self, alpha: np.ndarray, loop: np.ndarray, t: float) -> np.ndarray:
        n = alpha.size
        eps = self.values["eps"]
        half = (alpha[:, None] - alpha[None, :]) / 2.0  # (alpha_i - alpha_j)/2
        s, c = np.sin(half), np.cos(half)
        f = s * c / (s * s + eps * eps)  # cot_eps; the j = i diagonal is 0
        return self.values["w"] * self.omega + (self._k_matrix(loop) * f).sum(axis=1) / n


class ArcDynamics(Dynamics):
    """Per-arc local phases on a general curve graph (any number of
    crossings). RK4 on alpha with the arc labels frozen within the step
    (K_same / K_cross of the glued models distinguish same-arc from
    cross-arc pairs), then the stochastic vertex transition: passing
    alpha = 2*pi moves the oscillator uniformly onto one of the arcs
    leaving the end vertex; exiting backward through alpha = 0 moves it
    uniformly onto one of the arcs arriving at the start vertex. The
    phase overshoot is carried onto the new arc (alpha mod 2*pi)."""

    def __init__(
        self,
        model: GluedLoopModel,
        rng: np.random.Generator,
        forward_targets: list[np.ndarray],
        backward_targets: list[np.ndarray],
        stepper: Stepper | None = None,
    ) -> None:
        self.model = model
        self.rng = rng
        self.forward_targets = forward_targets
        self.backward_targets = backward_targets
        self.stepper: Stepper = stepper if stepper is not None else RK4Stepper()

    def step(self, state, t: float, dt: float):
        from ..space.arcs import ArcState

        labels = state.edge

        def f(alpha: np.ndarray, t_: float) -> np.ndarray:
            return self.model.dalpha(alpha, labels, t_)

        alpha = self.stepper.step(f, state.alpha, t, dt)
        new_edge = state.edge.copy()
        for i in np.flatnonzero(alpha >= TWO_PI):
            targets = self.forward_targets[int(state.edge[i])]
            new_edge[i] = targets[int(self.rng.integers(targets.size))]
        for i in np.flatnonzero(alpha < 0.0):
            targets = self.backward_targets[int(state.edge[i])]
            new_edge[i] = targets[int(self.rng.integers(targets.size))]
        return ArcState(new_edge, np.mod(alpha, TWO_PI))


class GluedDynamics(Dynamics):
    """RK4 on alpha (loops frozen within the step), then the stochastic
    boundary condition: any oscillator that passed the glue point alpha = 0
    in either direction has its loop redrawn uniformly from the m loops
    (probability 1/m each; the memo's 1/2-1/2 rule for m = 2)."""

    def __init__(
        self,
        model: GluedLoopModel,
        rng: np.random.Generator,
        n_loops: int = 2,
        stepper: Stepper | None = None,
    ) -> None:
        self.model = model
        self.rng = rng
        self.n_loops = int(n_loops)
        self.stepper: Stepper = stepper if stepper is not None else RK4Stepper()

    def step(self, state: GluedState, t: float, dt: float) -> GluedState:
        loop = state.loop

        def f(alpha: np.ndarray, t_: float) -> np.ndarray:
            return self.model.dalpha(alpha, loop, t_)

        alpha = self.stepper.step(f, state.alpha, t, dt)
        crossed = (alpha >= TWO_PI) | (alpha < 0.0)
        new_loop = loop.copy()
        if crossed.any():
            new_loop[crossed] = self.rng.integers(0, self.n_loops, size=int(crossed.sum()))
        return GluedState(new_loop, np.mod(alpha, TWO_PI))
