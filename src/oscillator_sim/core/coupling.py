"""Coupling functions g(d) for the graph mode (geodesic-distance based)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from ..registry import GRAPH_COUPLINGS


class GraphCoupling(ABC):
    """Speed correction kernel: (K/n) * sum_j g(d(x_i, x_j))."""

    name: ClassVar[str]

    @abstractmethod
    def g(self, d: np.ndarray, total_length: float) -> np.ndarray: ...


@GRAPH_COUPLINGS.register
class SineGeodesicCoupling(GraphCoupling):
    name = "sin(2*pi*d/L)"

    def g(self, d: np.ndarray, total_length: float) -> np.ndarray:
        return np.sin(2.0 * np.pi * d / total_length)
