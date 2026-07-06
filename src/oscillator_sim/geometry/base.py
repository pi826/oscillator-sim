"""Abstract embedded closed curve gamma: S1 -> R^2.

The parameter u lives in [0, 1); gamma must be smooth and closed. In S1
mode the curve is used for display and placement only; in graph mode its
image (with self-intersections as degree-4 vertices) defines the metric
graph the oscillators move on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np


class Curve(ABC):
    name: ClassVar[str]

    @abstractmethod
    def point(self, u: np.ndarray) -> np.ndarray:
        """Map parameters u (any shape) to points, shape u.shape + (2,)."""

    def polyline(self, samples: int) -> np.ndarray:
        """Open polyline of ``samples`` points at equidistant parameters."""
        u = np.linspace(0.0, 1.0, samples, endpoint=False)
        return self.point(u)
