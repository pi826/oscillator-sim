"""Branching rules applied when an oscillator reaches a graph vertex.

The candidates are all incidences at the vertex except the one the
oscillator arrived through (no immediate U-turn onto the same branch); at a
transversal self-intersection this leaves 3 choices.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from ..registry import BRANCHING_RULES
from ..space.graph import Incidence


class BranchingRule(ABC):
    name: ClassVar[str]

    @abstractmethod
    def choose(
        self,
        arrival: Incidence,
        candidates: list[Incidence],
        motion_dir: np.ndarray,
        rng: np.random.Generator,
    ) -> Incidence:
        """Pick the incidence to leave through.

        ``motion_dir`` is the unit direction of travel at the moment the
        vertex is reached (pointing into the vertex).
        """


@BRANCHING_RULES.register
class UniformRandomRule(BranchingRule):
    name = "Uniform random"

    def choose(self, arrival, candidates, motion_dir, rng):
        return candidates[int(rng.integers(len(candidates)))]


@BRANCHING_RULES.register
class StraightRule(BranchingRule):
    """Continue along the branch whose tangent is closest to the incoming
    direction of motion."""

    name = "Straight"

    def choose(self, arrival, candidates, motion_dir, rng):
        scores = [float(np.dot(motion_dir, c.out_tangent)) for c in candidates]
        return candidates[int(np.argmax(scores))]


@BRANCHING_RULES.register
class S1CompliantRule(BranchingRule):
    """Follow the original curve parameter (pass straight through the
    crossing as if it were not there)."""

    name = "S1-compliant"

    def choose(self, arrival, candidates, motion_dir, rng):
        # arriving at end 1 means traveling with increasing curve parameter:
        # continue onto the arc that *starts* (end 0) at the same parameter;
        # arriving at end 0 means decreasing parameter: continue onto the
        # arc that *ends* (end 1) there.
        wanted_end = 0 if arrival.end == 1 else 1
        best: Incidence | None = None
        best_gap = np.inf
        for c in candidates:
            if c.end != wanted_end:
                continue
            gap = abs(c.param - arrival.param) % 1.0
            gap = min(gap, 1.0 - gap)
            if gap < best_gap:
                best_gap = gap
                best = c
        if best is None:  # cannot happen for a graph built from a closed curve
            return candidates[0]
        return best
