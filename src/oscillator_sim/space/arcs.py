"""Per-arc local phases on the graph of any self-intersecting curve.

Generalizes the glued-loops construction to curves with several
crossings: the curve decomposes into arcs (the metric-graph edges); each
arc e carries a local phase alpha in [0, 2*pi) proportional to arclength,
with alpha = 0 at the arc's start vertex and alpha = 2*pi at its end
vertex. An oscillator whose phase passes 2*pi transitions stochastically
(uniformly) onto one of the arcs leaving that end vertex; exiting
backward through alpha = 0 transitions onto one of the arcs arriving at
the start vertex. On a bouquet curve (limacon, figure eight, rose) this
reduces exactly to the glued-loops mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import StateSpace
from .graph import MetricGraph

TWO_PI = 2.0 * np.pi


@dataclass
class ArcState:
    edge: np.ndarray  # (n,) int64 arc index
    alpha: np.ndarray  # (n,) float64 in [0, 2*pi)

    def copy(self) -> "ArcState":
        return ArcState(self.edge.copy(), self.alpha.copy())

    @property
    def n(self) -> int:
        return int(self.edge.size)


class ArcPhases(StateSpace):
    name = "Arc phases"
    placement_modes = ("uniform", "random")

    def __init__(self, curve, resolution: float = 1.0) -> None:
        graph = MetricGraph(curve, np.random.default_rng(0), resolution=resolution)
        self.curve = curve
        self._graph = graph
        self._edges = graph.edges
        self.n_arcs = len(graph.edges)

        out_of: dict[int, list[int]] = {}
        in_of: dict[int, list[int]] = {}
        for e in graph.edges:
            if e.v_start is not None:
                out_of.setdefault(e.v_start, []).append(e.index)
            if e.v_end is not None:
                in_of.setdefault(e.v_end, []).append(e.index)
        # a crossing-free closed loop (no vertices) wraps onto itself
        self.forward_targets = [
            np.array(out_of[e.v_end]) if e.v_end is not None else np.array([e.index])
            for e in graph.edges
        ]
        self.backward_targets = [
            np.array(in_of[e.v_start]) if e.v_start is not None else np.array([e.index])
            for e in graph.edges
        ]

    def polylines(self) -> list[np.ndarray]:
        return [edge.points for edge in self._edges]

    def arc_lengths(self) -> list[float]:
        return [edge.length for edge in self._edges]

    # --- StateSpace interface ------------------------------------------------

    def initial_states(self, n: int, rng: np.random.Generator, mode: str) -> ArcState:
        if mode == "uniform":
            alpha = TWO_PI * np.arange(n) / max(n, 1)
            edge = (np.arange(n) % self.n_arcs).astype(np.int64)
        elif mode == "random":
            alpha = rng.uniform(0.0, TWO_PI, size=n)
            edge = rng.integers(0, self.n_arcs, size=n).astype(np.int64)
        else:
            raise ValueError(f"unknown placement mode {mode!r}")
        return ArcState(edge, alpha)

    def positions(self, states: ArcState) -> np.ndarray:
        out = np.zeros((states.n, 2))
        for k, edge in enumerate(self._edges):
            mask = states.edge == k
            if mask.any():
                s = np.mod(states.alpha[mask], TWO_PI) / TWO_PI * edge.length
                out[mask] = edge.point_at(s)
        return out

    def add_at(self, states: ArcState, point: np.ndarray, rng: np.random.Generator) -> ArcState:
        graph = self._graph
        i = int(np.argmin(np.linalg.norm(graph._lookup - point[:2], axis=1)))
        edge = int(graph._lookup_edge[i])
        alpha = TWO_PI * float(graph._lookup_s[i]) / self._edges[edge].length
        return ArcState(
            np.append(states.edge, edge), np.append(states.alpha, alpha % TWO_PI)
        )

    def remove_index(self, states: ArcState, index: int) -> ArcState:
        return ArcState(np.delete(states.edge, index), np.delete(states.alpha, index))
