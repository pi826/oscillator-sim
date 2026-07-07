"""Metric graph built from a self-intersecting closed curve.

The image of gamma: S1 -> R^2 is viewed as a metric graph whose vertices
are the self-intersection points (degree 4 for a transversal crossing) and
whose edges are the arcs between them. An oscillator state is the record
(edge index, arclength coordinate s on the edge, heading sigma in {+1, -1}).

Geodesic distances: within an edge the arclength difference; across edges
the shortest path through vertices, with vertex-to-vertex distances
precomputed (Floyd-Warshall). The pair-distance distribution under the
uniform measure is presampled by Monte Carlo for the KS statistic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import (
    CURVE_ARCLEN_SAMPLES,
    CURVE_DETECT_SAMPLES,
    KS_MC_SAMPLES,
    VERTEX_MERGE_TOL,
)
from ..geometry.base import Curve
from .base import StateSpace


@dataclass
class GraphState:
    """State arrays for n oscillators on the graph."""

    edge: np.ndarray  # (n,) int64
    s: np.ndarray  # (n,) float64, 0 <= s <= edge length
    sigma: np.ndarray  # (n,) int64, +1 or -1

    def copy(self) -> "GraphState":
        return GraphState(self.edge.copy(), self.s.copy(), self.sigma.copy())

    @property
    def n(self) -> int:
        return int(self.edge.size)


@dataclass
class Edge:
    index: int
    v_start: int | None  # vertex at s = 0 (None only for a crossing-free loop)
    v_end: int | None  # vertex at s = length
    u_start: float  # curve parameter at s = 0
    u_end: float  # curve parameter at s = length (may exceed 1 when wrapping)
    points: np.ndarray  # (m, 2) polyline samples, s increasing with u
    arclens: np.ndarray  # (m,) cumulative arclength, arclens[0] = 0

    @property
    def length(self) -> float:
        return float(self.arclens[-1])

    def vertex(self, end: int) -> int | None:
        return self.v_start if end == 0 else self.v_end

    def point_at(self, s: np.ndarray | float) -> np.ndarray:
        x = np.interp(s, self.arclens, self.points[:, 0])
        y = np.interp(s, self.arclens, self.points[:, 1])
        return np.stack([x, y], axis=-1)

    def out_tangent(self, end: int) -> np.ndarray:
        """Unit tangent pointing from the endpoint into the edge."""
        d = self.points[1] - self.points[0] if end == 0 else self.points[-2] - self.points[-1]
        return d / np.linalg.norm(d)


@dataclass(frozen=True)
class Incidence:
    """One of the branch directions available at a vertex."""

    edge: int
    end: int  # 0: the edge leaves the vertex at s = 0, 1: at s = length
    vertex: int
    param: float  # curve parameter (mod 1) of this endpoint
    out_tangent: tuple[float, float]

    @property
    def entry_sigma(self) -> int:
        """Heading of an oscillator that leaves the vertex via this incidence."""
        return 1 if self.end == 0 else -1


def _detect_crossings(curve: Curve) -> list[tuple[float, float]]:
    """Find self-intersections; returns parameter pairs (u1, u2), u1 < u2."""
    n = CURVE_DETECT_SAMPLES
    u = np.arange(n) / n
    p = curve.point(u)
    q = p[(np.arange(n) + 1) % n]
    i_idx, j_idx = np.triu_indices(n, k=2)
    # segments 0 and n-1 are neighbors around the parameter wrap
    keep = ~((i_idx == 0) & (j_idx == n - 1))
    i_idx, j_idx = i_idx[keep], j_idx[keep]

    d1 = q[i_idx] - p[i_idx]
    d2 = q[j_idx] - p[j_idx]
    r = p[j_idx] - p[i_idx]
    denom = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (r[:, 0] * d2[:, 1] - r[:, 1] * d2[:, 0]) / denom
        w = (r[:, 0] * d1[:, 1] - r[:, 1] * d1[:, 0]) / denom
    # small slack so a crossing sitting exactly on a sample node (t or w
    # rounding to just below 0 / just below 1) is not missed
    eps = 1e-9
    hit = (np.abs(denom) > 1e-15) & (t >= -eps) & (t < 1) & (w >= -eps) & (w < 1)
    raw = [
        (float(((i + ti) / n) % 1.0), float(((j + wi) / n) % 1.0))
        for i, j, ti, wi in zip(i_idx[hit], j_idx[hit], t[hit], w[hit])
    ]

    # the slack can report one node crossing from several segment pairs:
    # deduplicate by circular closeness of the parameter pairs
    def circ(a: float, b: float) -> float:
        d = abs(a - b) % 1.0
        return min(d, 1.0 - d)

    tol = 2.0 / n
    kept: list[tuple[float, float]] = []
    for u1, u2 in raw:
        if any(
            (circ(u1, v1) < tol and circ(u2, v2) < tol)
            or (circ(u1, v2) < tol and circ(u2, v1) < tol)
            for v1, v2 in kept
        ):
            continue
        kept.append((u1, u2))
    return kept


class MetricGraph(StateSpace):
    name = "Metric graph"
    placement_modes = ("uniform", "random")

    def __init__(
        self, curve: Curve, rng: np.random.Generator, resolution: float = 1.0
    ) -> None:
        self.curve = curve
        self.resolution = max(float(resolution), 0.1)
        self._build(curve)
        self._presample_reference(rng)

    # --- construction ------------------------------------------------------

    def _build(self, curve: Curve) -> None:
        crossings = _detect_crossings(curve)

        # cluster crossing points into vertices
        pts = curve.point(np.array([c[0] for c in crossings])) if crossings else np.zeros((0, 2))
        diameter = float(np.ptp(curve.polyline(512), axis=0).max()) or 1.0
        tol = VERTEX_MERGE_TOL * diameter
        vertex_points: list[np.ndarray] = []
        breakpoints: list[tuple[float, int]] = []  # (param, vertex id)
        for (u1, u2), point in zip(crossings, pts):
            for k, vp in enumerate(vertex_points):
                if np.linalg.norm(vp - point) <= tol:
                    vid = k
                    break
            else:
                vid = len(vertex_points)
                vertex_points.append(point)
            breakpoints.append((u1, vid))
            breakpoints.append((u2, vid))
        self.vertex_points = np.array(vertex_points) if vertex_points else np.zeros((0, 2))
        self.n_vertices = len(vertex_points)

        # split the parameter circle into edges
        self.edges: list[Edge] = []
        if breakpoints:
            breakpoints.sort()
            for k, (u_lo, v_lo) in enumerate(breakpoints):
                if k + 1 < len(breakpoints):
                    u_hi, v_hi = breakpoints[k + 1]
                else:
                    u_hi, v_hi = breakpoints[0][0] + 1.0, breakpoints[0][1]
                self.edges.append(self._make_edge(len(self.edges), u_lo, u_hi, v_lo, v_hi))
        else:
            self.edges.append(self._make_edge(0, 0.0, 1.0, None, None))

        self.lengths = np.array([e.length for e in self.edges])
        self.total_length = float(self.lengths.sum())
        self._cum_lengths = np.concatenate([[0.0], np.cumsum(self.lengths)])
        # per-edge vertex ids as ints, -1 encoding "no vertex" (loop edge)
        self._edge_vertex_ids = np.array(
            [[-1 if e.v_start is None else e.v_start, -1 if e.v_end is None else e.v_end]
             for e in self.edges],
            dtype=np.int64,
        )

        # incidences per vertex
        self.incidences: list[list[Incidence]] = [[] for _ in range(self.n_vertices)]
        self._incidence_by_edge_end: dict[tuple[int, int], Incidence] = {}
        for e in self.edges:
            for end, vid, param in ((0, e.v_start, e.u_start), (1, e.v_end, e.u_end)):
                if vid is None:
                    continue
                inc = Incidence(
                    edge=e.index,
                    end=end,
                    vertex=vid,
                    param=float(param % 1.0),
                    out_tangent=tuple(e.out_tangent(end)),
                )
                self.incidences[vid].append(inc)
                self._incidence_by_edge_end[(e.index, end)] = inc

        # vertex-to-vertex shortest distances (Floyd-Warshall; V is tiny)
        v_count = self.n_vertices
        dist = np.full((v_count, v_count), np.inf)
        np.fill_diagonal(dist, 0.0)
        for e in self.edges:
            if e.v_start is not None and e.v_end is not None:
                a, b = e.v_start, e.v_end
                dist[a, b] = min(dist[a, b], e.length)
                dist[b, a] = dist[a, b]
        for k in range(v_count):
            dist = np.minimum(dist, dist[:, k : k + 1] + dist[k : k + 1, :])
        self.vertex_distances = dist

        # dense lookup for click placement
        self._lookup = np.vstack([e.points for e in self.edges])
        self._lookup_edge = np.concatenate(
            [np.full(len(e.points), e.index, dtype=np.int64) for e in self.edges]
        )
        self._lookup_s = np.concatenate([e.arclens for e in self.edges])

    def _make_edge(
        self, index: int, u_lo: float, u_hi: float, v_lo: int | None, v_hi: int | None
    ) -> Edge:
        m = max(8, int(np.ceil(CURVE_ARCLEN_SAMPLES * self.resolution * (u_hi - u_lo)))) + 1
        u = np.linspace(u_lo, u_hi, m)
        points = self.curve.point(np.mod(u, 1.0))
        seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
        arclens = np.concatenate([[0.0], np.cumsum(seg)])
        # snap endpoints of vertex-bounded edges onto the vertex point
        if v_lo is not None:
            points[0] = self.vertex_points[v_lo]
        if v_hi is not None:
            points[-1] = self.vertex_points[v_hi]
        return Edge(index, v_lo, v_hi, u_lo, u_hi, points, arclens)

    def incidence_of(self, edge: int, end: int) -> Incidence:
        return self._incidence_by_edge_end[(edge, end)]

    # --- geodesic distances ---------------------------------------------------

    def pairwise_distances(self, state: GraphState) -> np.ndarray:
        """Full (n, n) matrix of geodesic distances."""
        lengths = self.lengths[state.edge]
        ends = np.stack([state.s, lengths - state.s], axis=1)  # (n, 2)
        vids = self._edge_vertex_ids[state.edge]  # (n, 2)

        same = state.edge[:, None] == state.edge[None, :]
        direct = np.abs(state.s[:, None] - state.s[None, :])
        loop = self._edge_vertex_ids[state.edge][:, 0] == -1
        both_loop = same & loop[:, None]
        # a crossing-free closed loop wraps around
        direct = np.where(both_loop, np.minimum(direct, lengths[:, None] - direct), direct)

        via = np.full(direct.shape, np.inf)
        if self.n_vertices > 0:
            for a in (0, 1):
                for b in (0, 1):
                    va, vb = vids[:, a], vids[:, b]
                    ok = (va >= 0)[:, None] & (vb >= 0)[None, :]
                    vv = self.vertex_distances[np.maximum(va, 0)[:, None], np.maximum(vb, 0)[None, :]]
                    cand = ends[:, a][:, None] + vv + ends[:, b][None, :]
                    via = np.minimum(via, np.where(ok, cand, np.inf))
        return np.where(same, np.minimum(direct, via), via)

    def _distances_elementwise(
        self, edge1: np.ndarray, s1: np.ndarray, edge2: np.ndarray, s2: np.ndarray
    ) -> np.ndarray:
        """Geodesic distances between paired positions (for Monte Carlo)."""
        len1, len2 = self.lengths[edge1], self.lengths[edge2]
        ends1 = np.stack([s1, len1 - s1], axis=1)
        ends2 = np.stack([s2, len2 - s2], axis=1)
        v1, v2 = self._edge_vertex_ids[edge1], self._edge_vertex_ids[edge2]

        same = edge1 == edge2
        direct = np.abs(s1 - s2)
        loop = v1[:, 0] == -1
        direct = np.where(same & loop, np.minimum(direct, len1 - direct), direct)

        via = np.full(direct.shape, np.inf)
        if self.n_vertices > 0:
            for a in (0, 1):
                for b in (0, 1):
                    ok = (v1[:, a] >= 0) & (v2[:, b] >= 0)
                    vv = self.vertex_distances[np.maximum(v1[:, a], 0), np.maximum(v2[:, b], 0)]
                    cand = ends1[:, a] + vv + ends2[:, b]
                    via = np.minimum(via, np.where(ok, cand, np.inf))
        return np.where(same, np.minimum(direct, via), via)

    def _presample_reference(self, rng: np.random.Generator) -> None:
        """Pair-distance sample under the uniform measure (KS reference)."""
        m = KS_MC_SAMPLES
        e1, s1 = self.locate(rng.uniform(0.0, self.total_length, m))
        e2, s2 = self.locate(rng.uniform(0.0, self.total_length, m))
        self.reference_distances = np.sort(self._distances_elementwise(e1, s1, e2, s2))

    def sample_pair_distances(self, state: GraphState) -> np.ndarray:
        """All n(n-1)/2 current pair distances (for the KS statistic)."""
        d = self.pairwise_distances(state)
        iu = np.triu_indices(state.n, k=1)
        return d[iu]

    # --- coordinates --------------------------------------------------------

    def locate(self, global_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map global arclength in [0, L) to (edge index, local s)."""
        global_s = np.mod(np.asarray(global_s, dtype=np.float64), self.total_length)
        edge = np.clip(
            np.searchsorted(self._cum_lengths, global_s, side="right") - 1,
            0,
            len(self.edges) - 1,
        )
        return edge.astype(np.int64), global_s - self._cum_lengths[edge]

    # --- StateSpace interface -----------------------------------------------

    def initial_states(self, n: int, rng: np.random.Generator, mode: str) -> GraphState:
        if mode == "uniform":
            edge, s = self.locate((np.arange(n) + 0.5) * self.total_length / max(n, 1))
            sigma = np.ones(n, dtype=np.int64)
        elif mode == "random":
            edge, s = self.locate(rng.uniform(0.0, self.total_length, n))
            sigma = rng.choice(np.array([-1, 1], dtype=np.int64), size=n)
        else:
            raise ValueError(f"unknown placement mode {mode!r}")
        return GraphState(edge, s, sigma)

    def positions(self, states: GraphState) -> np.ndarray:
        out = np.zeros((states.n, 2))
        for e_idx in np.unique(states.edge):
            mask = states.edge == e_idx
            out[mask] = self.edges[int(e_idx)].point_at(states.s[mask])
        return out

    def add_at(self, states: GraphState, point: np.ndarray, rng: np.random.Generator) -> GraphState:
        i = int(np.argmin(np.linalg.norm(self._lookup - point[:2], axis=1)))
        sigma = int(rng.choice([-1, 1]))
        return GraphState(
            np.append(states.edge, self._lookup_edge[i]),
            np.append(states.s, self._lookup_s[i]),
            np.append(states.sigma, sigma),
        )

    def remove_index(self, states: GraphState, index: int) -> GraphState:
        return GraphState(
            np.delete(states.edge, index),
            np.delete(states.s, index),
            np.delete(states.sigma, index),
        )
