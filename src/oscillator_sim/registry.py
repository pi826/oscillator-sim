"""Generic name -> class registries.

Implementations register themselves with a decorator; GUI dropdowns are
built from registry contents, so adding a subclass is enough to expose it.
"""

from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A named collection of classes, keyed by their ``name`` attribute."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, cls: type[T]) -> type[T]:
        name = getattr(cls, "name", None)
        if not isinstance(name, str) or not name:
            raise ValueError(f"{cls!r} needs a non-empty class attribute 'name'")
        if name in self._items:
            raise ValueError(f"duplicate {self.kind} name: {name!r}")
        self._items[name] = cls
        return cls

    def names(self) -> list[str]:
        return list(self._items)

    def get(self, name: str) -> type[T]:
        return self._items[name]


MODELS: Registry = Registry("oscillator model")
SPHERE_MODELS: Registry = Registry("sphere model")
GLUED_MODELS: Registry = Registry("glued-loop model")
CURVES: Registry = Registry("curve")
BRANCHING_RULES: Registry = Registry("branching rule")
GRAPH_COUPLINGS: Registry = Registry("graph coupling")
