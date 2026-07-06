"""Abstraction of the space oscillators live in."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np


class StateSpace(ABC):
    """A state space with display embedding and mouse-placement support.

    ``states`` is opaque to callers: an angle array for the circle, a
    (edge, s, sigma) record for the metric graph, unit vectors for the
    sphere. All methods are functional (they return new state objects).
    """

    name: ClassVar[str]
    #: placement modes supported by ``initial_states``
    placement_modes: ClassVar[tuple[str, ...]] = ("uniform", "random")

    @abstractmethod
    def initial_states(self, n: int, rng: np.random.Generator, mode: str) -> Any: ...

    @abstractmethod
    def positions(self, states: Any) -> np.ndarray:
        """Display coordinates, shape (n, 2) or (n, 3)."""

    @abstractmethod
    def add_at(self, states: Any, point: np.ndarray, rng: np.random.Generator) -> Any:
        """Add one oscillator at the state nearest to ``point``."""

    @abstractmethod
    def remove_index(self, states: Any, index: int) -> Any: ...

    def count(self, states: Any) -> int:
        return int(self.positions(states).shape[0])

    def nearest_index(self, states: Any, point: np.ndarray) -> int:
        pos = self.positions(states)
        if pos.shape[0] == 0:
            raise ValueError("no oscillators")
        return int(np.argmin(np.linalg.norm(pos - point, axis=1)))
