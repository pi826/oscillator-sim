"""Abstract embedded closed curve gamma: S1 -> R^2.

The parameter u lives in [0, 1); gamma must be smooth and closed. In S1
mode the curve is used for display and placement only; in graph mode its
image (with self-intersections as degree-2k vertices) defines the metric
graph the oscillators move on.

Curves may declare ``params`` (ParamSpec entries, same mechanism as the
oscillator models): the GUI generates one spin box per entry and rebuilds
the simulation when a value changes, so curve families like the rose or
the trochoids are tunable without subclassing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from ..constants import CURVE_DETECT_SAMPLES
from ..core.params import Parameterized


class Curve(Parameterized, ABC):
    name: ClassVar[str]
    #: samples used for self-intersection detection in graph mode; curves
    #: with fine structure (many crossings) should raise this
    detect_samples: ClassVar[int] = CURVE_DETECT_SAMPLES

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def point(self, u: np.ndarray) -> np.ndarray:
        """Map parameters u (any shape) to points, shape u.shape + (2,)."""

    def polyline(self, samples: int) -> np.ndarray:
        """Open polyline of ``samples`` points at equidistant parameters."""
        u = np.linspace(0.0, 1.0, samples, endpoint=False)
        return self.point(u)
