"""Concrete curves; register a subclass to add it to the GUI dropdown."""

from __future__ import annotations

import numpy as np

from ..registry import CURVES
from .base import Curve

TWO_PI = 2.0 * np.pi


@CURVES.register
class UnitCircle(Curve):
    name = "Unit circle"

    def point(self, u: np.ndarray) -> np.ndarray:
        angle = TWO_PI * np.asarray(u)
        return np.stack([np.cos(angle), np.sin(angle)], axis=-1)


@CURVES.register
class Limacon(Curve):
    """Limacon r = b + a*cos(phi) with b < a: one self-intersection (origin)."""

    name = "Limacon"

    def __init__(self, a: float = 1.0, b: float = 0.5) -> None:
        if not b < a:
            raise ValueError("limacon needs b < a for an inner loop")
        self.a = a
        self.b = b

    def point(self, u: np.ndarray) -> np.ndarray:
        phi = TWO_PI * np.asarray(u)
        r = self.b + self.a * np.cos(phi)
        return np.stack([r * np.cos(phi), r * np.sin(phi)], axis=-1)


@CURVES.register
class FigureEight(Curve):
    """Lemniscate of Gerono: one transversal self-intersection at the origin."""

    name = "Figure eight"

    def point(self, u: np.ndarray) -> np.ndarray:
        s = TWO_PI * np.asarray(u)
        return np.stack([np.sin(s), np.sin(s) * np.cos(s)], axis=-1)
