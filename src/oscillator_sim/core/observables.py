"""Observables and state classification for all three state spaces.

Pure functions on NumPy arrays; all thresholds live in ``constants``.
"""

from __future__ import annotations

import numpy as np

from ..constants import (
    R_ONE_THRESHOLD,
    SPHERE_R_SYNC,
    SPHERE_R_ZERO,
    SPHERE_RING_TOL,
    SPHERE_UNIFORM_TOL,
)

TWO_PI = 2.0 * np.pi


# --- S1 ---------------------------------------------------------------------


def order_parameter(theta: np.ndarray, m: int = 1) -> float:
    """Generalized order parameter r_m = |sum_j exp(i m theta_j)| / n."""
    if theta.size == 0:
        return 0.0
    return float(np.abs(np.exp(1j * m * theta).mean()))


def order_parameters(theta: np.ndarray, m_max: int) -> np.ndarray:
    """r_m for m = 1..m_max as an array of length m_max."""
    ms = np.arange(1, m_max + 1)
    return np.abs(np.exp(1j * np.outer(ms, theta)).mean(axis=1))


def winding_number(theta: np.ndarray) -> int:
    """Winding of the phase sequence in index order around S1."""
    if theta.size < 2:
        return 0
    diffs = np.diff(theta, append=theta[:1])
    wrapped = np.mod(diffs + np.pi, TWO_PI) - np.pi
    return int(np.rint(wrapped.sum() / TWO_PI))


def classify_circle(theta: np.ndarray) -> str:
    """Classify an S1 configuration as sync / splay / (m,n)-pattern / incoherent.

    Uses r_m (m = 1..n) together with the winding number w of the phase
    order: q equally spaced phase values give r_q ~ 1, so the smallest such
    q identifies the pattern; q = 1 is sync, q = n with |w| = 1 is splay,
    any other q is reported as a (w, q)-pattern, otherwise incoherent.
    """
    n = theta.size
    if n < 2:
        return "-"
    r = order_parameters(theta, n)
    near_one = np.flatnonzero(r >= R_ONE_THRESHOLD)
    if near_one.size == 0:
        return "incoherent"
    q = int(near_one[0]) + 1
    if q == 1:
        return "sync"
    w = winding_number(theta)
    if q == n and abs(w) == 1:
        return "splay"
    return f"({w},{q})-pattern"


# --- metric graph -----------------------------------------------------------


def uniformity(edge_index: np.ndarray, edge_lengths: np.ndarray, total_length: float) -> float:
    """U(t) = 1 - (1/2) sum_e |rho_e - l_e / L| (1 = uniform occupation)."""
    n = edge_index.size
    if n == 0:
        return 0.0
    counts = np.bincount(edge_index, minlength=edge_lengths.size).astype(np.float64)
    rho = counts / n
    ell = edge_lengths / total_length
    return float(1.0 - 0.5 * np.abs(rho - ell).sum())


def ks_statistic(sample: np.ndarray, reference: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic sup |F_sample - F_reference|.

    ``reference`` is the (large, precomputed) sample of pair distances under
    the uniform distribution on the graph.
    """
    if sample.size == 0 or reference.size == 0:
        return 0.0
    sample = np.sort(sample)
    reference = np.sort(reference)
    grid = np.concatenate([sample, reference])
    cdf_s = np.searchsorted(sample, grid, side="right") / sample.size
    cdf_r = np.searchsorted(reference, grid, side="right") / reference.size
    return float(np.abs(cdf_s - cdf_r).max())


def edge_polarization(edge_index: np.ndarray, sigma: np.ndarray, n_edges: int) -> np.ndarray:
    """p_e = (#sigma=+1 - #sigma=-1) / (#oscillators on e); 0 for empty edges."""
    totals = np.bincount(edge_index, minlength=n_edges).astype(np.float64)
    signed = np.bincount(edge_index, weights=sigma.astype(np.float64), minlength=n_edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(totals > 0, signed / totals, 0.0)
    return p


# --- sphere -----------------------------------------------------------------


def sphere_order_parameter(x: np.ndarray) -> float:
    """r = ||(1/n) sum_j x_j||."""
    if x.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(x.mean(axis=0)))


def classify_sphere(x: np.ndarray) -> str:
    """Classify via r and the eigenvalues of S = (1/n) sum_j x_j x_j^T.

    S2 is simply connected (no winding number), so the S1 (m,n)-pattern
    taxonomy does not apply; instead: sync (r ~ 1), uniform (r ~ 0 and
    S ~ I/3), ring (r ~ 0 and eigenvalues ~ (1/2, 1/2, 0)), else clustered.
    """
    n = x.shape[0]
    if n < 2:
        return "-"
    r = sphere_order_parameter(x)
    if r >= SPHERE_R_SYNC:
        return "sync"
    s_matrix = x.T @ x / n
    if r <= SPHERE_R_ZERO:
        if np.linalg.norm(s_matrix - np.eye(3) / 3.0) <= SPHERE_UNIFORM_TOL:
            return "uniform"
        eig = np.sort(np.linalg.eigvalsh(s_matrix))[::-1]
        if np.linalg.norm(eig - np.array([0.5, 0.5, 0.0])) <= SPHERE_RING_TOL:
            return "ring"
    return "clustered"
