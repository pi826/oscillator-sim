"""The unit sphere S2 in R^3: states are unit row vectors, shape (n, 3)."""

from __future__ import annotations

import numpy as np

from .base import StateSpace


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class Sphere(StateSpace):
    name = "Sphere (S2)"
    placement_modes = ("random",)

    def initial_states(self, n: int, rng: np.random.Generator, mode: str) -> np.ndarray:
        if mode != "random":
            raise ValueError(f"unknown placement mode {mode!r}")
        # normalized Gaussians are exactly uniform on the sphere
        return _normalize_rows(rng.normal(size=(n, 3)))

    def positions(self, states: np.ndarray) -> np.ndarray:
        return states

    def add_at(self, states: np.ndarray, point: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        p = np.asarray(point, dtype=np.float64)
        p = p / np.linalg.norm(p)
        return np.vstack([states, p[None, :]])

    def remove_index(self, states: np.ndarray, index: int) -> np.ndarray:
        return np.delete(states, index, axis=0)
