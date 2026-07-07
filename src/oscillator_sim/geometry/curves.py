"""Concrete curves; register a subclass to add it to the GUI dropdown.

Curves with ``params`` get auto-generated spin boxes in the GUI (integer
parameters use ``decimals=0``; non-integer winding numbers would break the
closure gamma(0) = gamma(1)).
"""

from __future__ import annotations

import numpy as np

from ..core.params import ParamSpec
from ..registry import CURVES
from .base import Curve

TWO_PI = 2.0 * np.pi


@CURVES.register
class UnitCircle(Curve):
    name = "Unit circle"

    def point(self, u: np.ndarray) -> np.ndarray:
        angle = TWO_PI * np.asarray(u)
        return np.stack([np.cos(angle), np.sin(angle)], axis=-1)


@CURVES.register
class Limacon(Curve):
    """Limacon r = b + cos(phi): one self-intersection (origin) for b < 1,
    a cusp at b = 1, convex-ish and crossing-free for b > 1."""

    name = "Limacon"
    params = {"b": ParamSpec("b (offset)", 0.5, 0.05, 2.0, 0.05)}

    def point(self, u: np.ndarray) -> np.ndarray:
        phi = TWO_PI * np.asarray(u)
        r = self.values["b"] + np.cos(phi)
        return np.stack([r * np.cos(phi), r * np.sin(phi)], axis=-1)


@CURVES.register
class FigureEight(Curve):
    """Lemniscate of Gerono: one transversal self-intersection at the origin."""

    name = "Figure eight"

    def point(self, u: np.ndarray) -> np.ndarray:
        s = TWO_PI * np.asarray(u)
        return np.stack([np.sin(s), np.sin(s) * np.cos(s)], axis=-1)


@CURVES.register
class Rose(Curve):
    """Rose r = cos(k*phi). Odd k: k petals (phi in [0, pi) traces the
    whole curve once); even k: 2k petals over [0, 2*pi). All petals meet
    at the origin, which becomes a single high-degree vertex in graph mode.
    """

    name = "Rose"
    params = {"k": ParamSpec("k (petal factor)", 3, 2, 8, 1, decimals=0)}
    detect_samples = 2048

    def point(self, u: np.ndarray) -> np.ndarray:
        k = int(round(self.values["k"]))
        span = np.pi if k % 2 == 1 else TWO_PI
        phi = span * np.asarray(u)
        r = np.cos(k * phi)
        return np.stack([r * np.cos(phi), r * np.sin(phi)], axis=-1)


@CURVES.register
class Astroid(Curve):
    """Astroid x = cos^3 t, y = sin^3 t: four cusps, no self-intersection."""

    name = "Astroid"

    def point(self, u: np.ndarray) -> np.ndarray:
        s = TWO_PI * np.asarray(u)
        return np.stack([np.cos(s) ** 3, np.sin(s) ** 3], axis=-1)


@CURVES.register
class Hypotrochoid(Curve):
    """Spirograph inside a ring (R = k+1, r = 1):
    x = k cos t + d cos(k t), y = k sin t - d sin(k t).
    d > 1 creates k+1 loops / self-intersections; d = 1 is the cusped
    hypocycloid (deltoid for k = 2, astroid for k = 3)."""

    name = "Hypotrochoid"
    params = {
        "k": ParamSpec("k (ratio)", 2, 2, 9, 1, decimals=0),
        "d": ParamSpec("d (pen offset)", 1.5, 0.1, 3.0, 0.1),
    }
    detect_samples = 2048

    def point(self, u: np.ndarray) -> np.ndarray:
        k = int(round(self.values["k"]))
        d = self.values["d"]
        s = TWO_PI * np.asarray(u)
        x = k * np.cos(s) + d * np.cos(k * s)
        y = k * np.sin(s) - d * np.sin(k * s)
        return np.stack([x, y], axis=-1) / (k + d)


@CURVES.register
class Epitrochoid(Curve):
    """Spirograph outside a ring (R = k-1, r = 1):
    x = k cos t - d cos(k t), y = k sin t - d sin(k t).
    d > 1 creates k-1 inner loops / self-intersections."""

    name = "Epitrochoid"
    params = {
        "k": ParamSpec("k (ratio)", 3, 2, 9, 1, decimals=0),
        "d": ParamSpec("d (pen offset)", 2.0, 0.1, 3.0, 0.1),
    }
    detect_samples = 2048

    def point(self, u: np.ndarray) -> np.ndarray:
        k = int(round(self.values["k"]))
        d = self.values["d"]
        s = TWO_PI * np.asarray(u)
        x = k * np.cos(s) - d * np.cos(k * s)
        y = k * np.sin(s) - d * np.sin(k * s)
        return np.stack([x, y], axis=-1) / (k + d)


@CURVES.register
class Trefoil(Curve):
    """Hypotrochoid with k = 2 kept as a named preset: three loops, three
    transversal self-intersections (a planar trefoil shadow)."""

    name = "Trefoil"
    params = {"d": ParamSpec("d (pen offset)", 1.5, 0.1, 3.0, 0.1)}

    def point(self, u: np.ndarray) -> np.ndarray:
        d = self.values["d"]
        s = TWO_PI * np.asarray(u)
        x = 2.0 * np.cos(s) + d * np.cos(2.0 * s)
        y = 2.0 * np.sin(s) - d * np.sin(2.0 * s)
        return 0.4 * np.stack([x, y], axis=-1)


@CURVES.register
class TrefoilKnot(Curve):
    """Planar projection of the trefoil knot (the (2,3) torus knot):
    x = sin t + 2 sin 2t, y = cos t - 2 cos 2t; three crossings."""

    name = "Trefoil knot (projection)"

    def point(self, u: np.ndarray) -> np.ndarray:
        s = TWO_PI * np.asarray(u)
        x = np.sin(s) + 2.0 * np.sin(2.0 * s)
        y = np.cos(s) - 2.0 * np.cos(2.0 * s)
        return np.stack([x, y], axis=-1) / 3.0


@CURVES.register
class Lissajous(Curve):
    """Lissajous curve x = sin(p t + delta), y = sin(q t). p and q should
    be coprime (otherwise the curve retraces itself)."""

    name = "Lissajous"
    params = {
        "p": ParamSpec("p (x frequency)", 3, 1, 9, 1, decimals=0),
        "q": ParamSpec("q (y frequency)", 2, 1, 9, 1, decimals=0),
        "delta": ParamSpec("delta (x phase)", 0.0, -1.6, 1.6, 0.05),
    }
    detect_samples = 2048

    def point(self, u: np.ndarray) -> np.ndarray:
        p = int(round(self.values["p"]))
        q = int(round(self.values["q"]))
        s = TWO_PI * np.asarray(u)
        return np.stack([np.sin(p * s + self.values["delta"]), np.sin(q * s)], axis=-1)


@CURVES.register
class Butterfly(Curve):
    """Fay's butterfly r = e^{sin t} - 2 cos 4t + sin^5((2t - pi)/24),
    closed over t in [0, 24*pi); dozens of self-intersections."""

    name = "Butterfly (Fay)"
    detect_samples = 4096

    def point(self, u: np.ndarray) -> np.ndarray:
        t = 24.0 * np.pi * np.asarray(u)
        r = np.exp(np.sin(t)) - 2.0 * np.cos(4.0 * t) + np.sin((2.0 * t - np.pi) / 24.0) ** 5
        return np.stack([r * np.sin(t), r * np.cos(t)], axis=-1) / 4.0


@CURVES.register
class Superellipse(Curve):
    """Superellipse |x|^p + |y/b|^p = 1: p = 2 ellipse, p = 4 squircle,
    p < 1 concave star; no self-intersections."""

    name = "Superellipse"
    params = {
        "p": ParamSpec("p (exponent)", 4.0, 0.5, 8.0, 0.1),
        "b": ParamSpec("b (aspect)", 0.7, 0.2, 1.0, 0.05),
    }

    def point(self, u: np.ndarray) -> np.ndarray:
        s = TWO_PI * np.asarray(u)
        e = 2.0 / self.values["p"]
        c, si = np.cos(s), np.sin(s)
        x = np.abs(c) ** e * np.sign(c)
        y = self.values["b"] * np.abs(si) ** e * np.sign(si)
        return np.stack([x, y], axis=-1)


@CURVES.register
class Heart(Curve):
    """The classic heart curve; no self-intersections."""

    name = "Heart"

    def point(self, u: np.ndarray) -> np.ndarray:
        s = TWO_PI * np.asarray(u)
        x = 16.0 * np.sin(s) ** 3
        y = 13.0 * np.cos(s) - 5.0 * np.cos(2.0 * s) - 2.0 * np.cos(3.0 * s) - np.cos(4.0 * s)
        return np.stack([x, y], axis=-1) / 17.0
