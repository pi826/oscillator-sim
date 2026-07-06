"""The standard circle S1: states are phases theta in [0, 2*pi).

An embedded curve gamma: S1 -> R^2 (from the geometry layer) is used for
display and mouse placement only; the dynamics never sees it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import StateSpace

if TYPE_CHECKING:
    from ..geometry.base import Curve

TWO_PI = 2.0 * np.pi

# resolution of the curve lookup table used for display / click placement
_LOOKUP_SAMPLES = 2048


class Circle(StateSpace):
    name = "Circle (S1)"
    placement_modes = ("uniform", "random", "splay")

    def __init__(self, curve: "Curve | None" = None) -> None:
        self._curve: "Curve | None" = None
        self._lookup_u = np.linspace(0.0, 1.0, _LOOKUP_SAMPLES, endpoint=False)
        self._lookup_xy: np.ndarray | None = None
        self.set_curve(curve)

    def set_curve(self, curve: "Curve | None") -> None:
        self._curve = curve
        if curve is not None:
            self._lookup_xy = curve.point(self._lookup_u)
        else:
            self._lookup_xy = None

    def curve_polyline(self, samples: int = 720) -> np.ndarray:
        """Closed polyline of the display curve, shape (samples + 1, 2)."""
        u = np.linspace(0.0, 1.0, samples, endpoint=False)
        pts = self._embed(u)
        return np.vstack([pts, pts[:1]])

    def _embed(self, u: np.ndarray) -> np.ndarray:
        if self._curve is not None:
            return self._curve.point(u)
        angle = TWO_PI * u
        return np.column_stack([np.cos(angle), np.sin(angle)])

    # --- StateSpace interface -------------------------------------------

    def initial_states(self, n: int, rng: np.random.Generator, mode: str) -> np.ndarray:
        if mode == "random":
            return rng.uniform(0.0, TWO_PI, size=n)
        if mode in ("uniform", "splay"):
            # splay placement theta_j = 2*pi*j/n; on S1 this coincides with
            # the deterministic uniform placement
            return TWO_PI * np.arange(n) / max(n, 1)
        raise ValueError(f"unknown placement mode {mode!r}")

    def positions(self, states: np.ndarray) -> np.ndarray:
        return self._embed(np.mod(states, TWO_PI) / TWO_PI)

    def add_at(self, states: np.ndarray, point: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self._lookup_xy is not None:
            i = int(np.argmin(np.linalg.norm(self._lookup_xy - point[:2], axis=1)))
            theta = TWO_PI * self._lookup_u[i]
        else:
            theta = float(np.arctan2(point[1], point[0])) % TWO_PI
        return np.append(states, theta)

    def remove_index(self, states: np.ndarray, index: int) -> np.ndarray:
        return np.delete(states, index)
