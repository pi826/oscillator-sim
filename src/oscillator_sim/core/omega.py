"""Generation of natural frequencies (S1 / graph) and rotations (sphere).

All randomness is drawn from the caller's ``numpy.random.Generator`` so a
single seed reproduces the whole simulation.
"""

from __future__ import annotations

import numpy as np

from ..constants import OMEGA_MEAN, OMEGA_STD, SPHERE_OMEGA_MEAN, SPHERE_OMEGA_STD

OMEGA_MODES: list[str] = ["Identical", "Normal"]

SPHERE_ROTATION_MODES: list[str] = [
    "All zero",
    "Common axis + normal speeds",
    "Random axes and speeds",
]


def make_omega(mode: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Natural frequencies omega_i for S1 and graph oscillators."""
    if mode == "Identical":
        return np.full(n, OMEGA_MEAN)
    if mode == "Normal":
        return rng.normal(OMEGA_MEAN, OMEGA_STD, size=n)
    raise ValueError(f"unknown omega mode {mode!r}")


def make_rotations(mode: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Angular-velocity vectors w_i (shape (n, 3)) for the sphere mode.

    The skew-symmetric matrix Omega_i in the Lohe model acts on x as
    Omega_i x = w_i x x (cross product), so storing w_i is equivalent.
    """
    if mode == "All zero":
        return np.zeros((n, 3))
    if mode == "Common axis + normal speeds":
        speeds = rng.normal(SPHERE_OMEGA_MEAN, SPHERE_OMEGA_STD, size=n)
        return np.outer(speeds, np.array([0.0, 0.0, 1.0]))
    if mode == "Random axes and speeds":
        axes = rng.normal(size=(n, 3))
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        speeds = rng.normal(SPHERE_OMEGA_MEAN, SPHERE_OMEGA_STD, size=n)
        return axes * speeds[:, None]
    raise ValueError(f"unknown rotation mode {mode!r}")
