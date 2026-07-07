"""Two loops of a self-intersecting limacon glued at the crossing point.

Following the limacon_branching_kuramoto memo: the limacon r = b + cos(phi)
(b < 1) splits at its self-intersection into an inner and an outer loop.
Rather than the planar curve, the state space is the pair

    (loop in {inner=0, outer=1}, local phase alpha in [0, 2*pi))

where alpha runs once around the respective loop and alpha = 0 is the
crossing point on both loops. For display, alpha is mapped to the loop
proportionally to arclength, so equal phase speed looks uniform on screen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import StateSpace

TWO_PI = 2.0 * np.pi

# maximum b: beyond 1 the inner loop (and the crossing) disappears
_B_MAX = 0.95
_ARC_SAMPLES = 1024  # per loop, before the resolution multiplier


@dataclass
class GluedState:
    loop: np.ndarray  # (n,) int64, 0 = inner, 1 = outer
    alpha: np.ndarray  # (n,) float64 in [0, 2*pi)

    def copy(self) -> "GluedState":
        return GluedState(self.loop.copy(), self.alpha.copy())

    @property
    def n(self) -> int:
        return int(self.loop.size)


class _Arc:
    """Arclength-parameterized piece of the limacon."""

    def __init__(self, curve, u_lo: float, u_hi: float, samples: int) -> None:
        u = np.linspace(u_lo, u_hi, samples)
        self.points = curve.point(np.mod(u, 1.0))
        seg = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.arclens = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.arclens[-1])

    def at(self, alpha: np.ndarray) -> np.ndarray:
        s = np.mod(alpha, TWO_PI) / TWO_PI * self.length
        x = np.interp(s, self.arclens, self.points[:, 0])
        y = np.interp(s, self.arclens, self.points[:, 1])
        return np.stack([x, y], axis=-1)


class GluedLoops(StateSpace):
    name = "Glued loops (Limacon)"
    placement_modes = ("uniform", "random")

    def __init__(self, curve, resolution: float = 1.0) -> None:
        """``curve`` must be a Limacon-style curve with a ``b`` parameter;
        b is clamped below 1 so the self-intersection exists."""
        b = min(float(curve.values["b"]), _B_MAX)
        curve.set_param("b", b)
        self.curve = curve
        # r = b + cos(phi) vanishes at phi0 = arccos(-b); the inner loop is
        # the r < 0 stretch phi in (phi0, 2*pi - phi0)
        u0 = float(np.arccos(-b)) / TWO_PI
        m = max(64, int(_ARC_SAMPLES * resolution))
        self._arcs = [
            _Arc(curve, u0, 1.0 - u0, m),  # 0: inner
            _Arc(curve, 1.0 - u0, 1.0 + u0, m),  # 1: outer
        ]

    def polylines(self) -> list[np.ndarray]:
        return [arc.points for arc in self._arcs]

    def loop_lengths(self) -> tuple[float, float]:
        return self._arcs[0].length, self._arcs[1].length

    # --- StateSpace interface ------------------------------------------------

    def initial_states(self, n: int, rng: np.random.Generator, mode: str) -> GluedState:
        if mode == "uniform":
            alpha = TWO_PI * np.arange(n) / max(n, 1)
            loop = (np.arange(n) % 2).astype(np.int64)
        elif mode == "random":
            alpha = rng.uniform(0.0, TWO_PI, size=n)
            loop = rng.integers(0, 2, size=n).astype(np.int64)
        else:
            raise ValueError(f"unknown placement mode {mode!r}")
        return GluedState(loop, alpha)

    def positions(self, states: GluedState) -> np.ndarray:
        out = np.zeros((states.n, 2))
        for k in (0, 1):
            mask = states.loop == k
            if mask.any():
                out[mask] = self._arcs[k].at(states.alpha[mask])
        return out

    def add_at(self, states: GluedState, point: np.ndarray, rng: np.random.Generator) -> GluedState:
        best: tuple[float, int, float] | None = None
        for k, arc in enumerate(self._arcs):
            d = np.linalg.norm(arc.points - point[:2], axis=1)
            i = int(np.argmin(d))
            alpha = TWO_PI * arc.arclens[i] / arc.length
            if best is None or d[i] < best[0]:
                best = (float(d[i]), k, float(alpha % TWO_PI))
        assert best is not None
        return GluedState(
            np.append(states.loop, best[1]), np.append(states.alpha, best[2])
        )

    def remove_index(self, states: GluedState, index: int) -> GluedState:
        return GluedState(np.delete(states.loop, index), np.delete(states.alpha, index))
