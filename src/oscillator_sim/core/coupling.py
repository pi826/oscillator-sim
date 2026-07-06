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


@GRAPH_COUPLINGS.register
class RepulsiveSineCoupling(GraphCoupling):
    """Sign-flipped kernel: nearby oscillators slow each other down less
    than distant ones, which spreads the population out (U -> 1)."""

    name = "-sin(2*pi*d/L)"

    def g(self, d: np.ndarray, total_length: float) -> np.ndarray:
        return -np.sin(2.0 * np.pi * d / total_length)


@GRAPH_COUPLINGS.register
class SecondHarmonicCoupling(GraphCoupling):
    """Second-harmonic kernel; favors two antipodal packs on the graph."""

    name = "sin(4*pi*d/L)"

    def g(self, d: np.ndarray, total_length: float) -> np.ndarray:
        return np.sin(4.0 * np.pi * d / total_length)


@GRAPH_COUPLINGS.register
class MexicanHatCoupling(GraphCoupling):
    """Difference of Gaussians (short-range boost, mid-range damping):
    g(d) = exp(-(d/(s1*L))^2) - c * exp(-(d/(s2*L))^2). Tends to organize
    the population into evenly spaced pulse packs."""

    name = "Mexican hat"
    SIGMA_NEAR = 0.1
    SIGMA_FAR = 0.3
    DEPTH = 0.5

    def g(self, d: np.ndarray, total_length: float) -> np.ndarray:
        near = np.exp(-((d / (self.SIGMA_NEAR * total_length)) ** 2))
        far = np.exp(-((d / (self.SIGMA_FAR * total_length)) ** 2))
        return near - self.DEPTH * far
