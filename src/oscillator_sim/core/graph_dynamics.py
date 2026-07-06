"""Dynamics of oscillators moving freely on a metric graph.

Each oscillator moves with intrinsic speed omega_i along its heading
sigma_i, corrected by the geodesic coupling (K/n) sum_j g(d(x_i, x_j)).
The speed field is integrated with RK4 (stage positions advance within
their current edge, clamped at vertices); the resulting displacement is
then applied event-wise: motion is cut exactly at each vertex, the
branching rule picks the next branch, and the remaining distance is
carried onto it, so no crossing error is left behind.
"""

from __future__ import annotations

import numpy as np

from ..constants import MAX_CROSSINGS_PER_STEP
from ..space.graph import GraphState, MetricGraph
from .branching import BranchingRule
from .coupling import GraphCoupling
from .params import Parameterized, ParamSpec
from .simulation import Dynamics


class GraphDynamics(Dynamics, Parameterized):
    params = {"K": ParamSpec("K (coupling)", 1.0, -10.0, 10.0, 0.1)}

    def __init__(
        self,
        graph: MetricGraph,
        omega: np.ndarray,
        coupling: GraphCoupling,
        branching: BranchingRule,
        rng: np.random.Generator,
    ) -> None:
        Parameterized.__init__(self)
        self.graph = graph
        self.omega = np.asarray(omega, dtype=np.float64)
        self.coupling = coupling
        self.branching = branching
        self.rng = rng

    def speeds(self, state: GraphState) -> np.ndarray:
        """Scalar speed along the heading: omega_i + (K/n) sum_j g(d_ij)."""
        n = state.n
        d = self.graph.pairwise_distances(state)
        g = self.coupling.g(d, self.graph.total_length)
        np.fill_diagonal(g, 0.0)
        return self.omega + (self.values["K"] / n) * g.sum(axis=1)

    def step(self, state: GraphState, t: float, dt: float) -> GraphState:
        k1 = self.speeds(state)
        k2 = self.speeds(self._advance_clamped(state, state.sigma * k1 * (0.5 * dt)))
        k3 = self.speeds(self._advance_clamped(state, state.sigma * k2 * (0.5 * dt)))
        k4 = self.speeds(self._advance_clamped(state, state.sigma * k3 * dt))
        speed = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return self._move(state, state.sigma * speed * dt)

    # --- helpers -----------------------------------------------------------

    def _advance_clamped(self, state: GraphState, disp: np.ndarray) -> GraphState:
        """Stage positions for RK4: move within the current edge only."""
        new = state.copy()
        new.s = np.clip(new.s + disp, 0.0, self.graph.lengths[new.edge])
        return new

    def _move(self, state: GraphState, disp: np.ndarray) -> GraphState:
        new = state.copy()
        lengths = self.graph.lengths[new.edge]
        target = new.s + disp
        inside = (target >= 0.0) & (target <= lengths)
        new.s[inside] = target[inside]
        for i in np.flatnonzero(~inside):
            self._move_one(new, int(i), float(disp[i]))
        return new

    def _move_one(self, state: GraphState, i: int, disp: float) -> None:
        """Event-driven motion of one oscillator across vertices."""
        graph = self.graph
        e = int(state.edge[i])
        s = float(state.s[i])
        direction = 1 if disp > 0 else -1  # along the s-axis of the edge
        remaining = abs(disp)

        for _ in range(MAX_CROSSINGS_PER_STEP):
            edge = graph.edges[e]
            room = edge.length - s if direction > 0 else s
            if remaining <= room:
                s += direction * remaining
                break
            remaining -= room
            end = 1 if direction > 0 else 0
            if edge.vertex(end) is None:
                # crossing-free closed loop: wrap around
                s = 0.0 if direction > 0 else edge.length
                continue
            arrival = graph.incidence_of(e, end)
            motion_dir = -np.asarray(arrival.out_tangent)
            candidates = [c for c in graph.incidences[arrival.vertex] if c is not arrival]
            chosen = self.branching.choose(arrival, candidates, motion_dir, self.rng)
            e = chosen.edge
            if chosen.end == 0:
                s, direction = 0.0, 1
            else:
                s, direction = graph.edges[e].length, -1
        else:
            raise RuntimeError(
                "oscillator crossed too many vertices in one step; reduce dt or speed"
            )

        state.edge[i] = e
        state.s[i] = s
        state.sigma[i] = direction
