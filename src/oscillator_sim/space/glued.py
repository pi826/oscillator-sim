"""Loops of a self-intersecting curve glued at a single crossing point.

Generalizes the limacon_branching_kuramoto memo: any curve whose
self-intersections form ONE vertex through which every arc closes into a
loop (a bouquet of circles) qualifies - the limacon (b < 1, two loops),
the figure eight (two loops), and the rose (k petals for odd k, 2k for
even k). The state is

    (loop in {0..m-1}, local phase alpha in [0, 2*pi))

where alpha runs once around the respective loop and alpha = 0 is the
common crossing point on every loop. For display, alpha is mapped to the
loop proportionally to arclength.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import StateSpace
from .graph import MetricGraph

TWO_PI = 2.0 * np.pi


@dataclass
class GluedState:
    loop: np.ndarray  # (n,) int64 in {0..m-1}
    alpha: np.ndarray  # (n,) float64 in [0, 2*pi)

    def copy(self) -> "GluedState":
        return GluedState(self.loop.copy(), self.alpha.copy())

    @property
    def n(self) -> int:
        return int(self.loop.size)


class GluedLoops(StateSpace):
    name = "Glued loops"
    placement_modes = ("uniform", "random")

    def __init__(self, curve, resolution: float = 1.0) -> None:
        graph = MetricGraph(curve, np.random.default_rng(0), resolution=resolution)
        if graph.n_vertices != 1 or any(e.v_start != e.v_end for e in graph.edges):
            raise ValueError(
                "glued-loops mode needs a curve whose self-intersections form a "
                "single point with every arc closing into a loop through it "
                "(e.g. Limacon with b < 1, Figure eight, Rose)"
            )
        self.curve = curve
        self._graph = graph
        self._edges = graph.edges
        self.n_loops = len(graph.edges)

    def polylines(self) -> list[np.ndarray]:
        return [edge.points for edge in self._edges]

    def loop_lengths(self) -> list[float]:
        return [edge.length for edge in self._edges]

    # --- StateSpace interface ------------------------------------------------

    def initial_states(self, n: int, rng: np.random.Generator, mode: str) -> GluedState:
        if mode == "uniform":
            alpha = TWO_PI * np.arange(n) / max(n, 1)
            loop = (np.arange(n) % self.n_loops).astype(np.int64)
        elif mode == "random":
            alpha = rng.uniform(0.0, TWO_PI, size=n)
            loop = rng.integers(0, self.n_loops, size=n).astype(np.int64)
        else:
            raise ValueError(f"unknown placement mode {mode!r}")
        return GluedState(loop, alpha)

    def positions(self, states: GluedState) -> np.ndarray:
        out = np.zeros((states.n, 2))
        for k, edge in enumerate(self._edges):
            mask = states.loop == k
            if mask.any():
                s = np.mod(states.alpha[mask], TWO_PI) / TWO_PI * edge.length
                out[mask] = edge.point_at(s)
        return out

    def add_at(self, states: GluedState, point: np.ndarray, rng: np.random.Generator) -> GluedState:
        graph = self._graph
        i = int(np.argmin(np.linalg.norm(graph._lookup - point[:2], axis=1)))
        loop = int(graph._lookup_edge[i])
        alpha = TWO_PI * float(graph._lookup_s[i]) / self._edges[loop].length
        return GluedState(
            np.append(states.loop, loop), np.append(states.alpha, alpha % TWO_PI)
        )

    def remove_index(self, states: GluedState, index: int) -> GluedState:
        return GluedState(np.delete(states.loop, index), np.delete(states.alpha, index))
