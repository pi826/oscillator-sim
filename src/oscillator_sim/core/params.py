"""Declarative model parameters.

Models declare their parameters as ``ParamSpec`` entries; the GUI generates
one spin box per entry, so new models need no GUI code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class ParamSpec:
    label: str
    default: float
    minimum: float
    maximum: float
    step: float
    #: spin-box decimals; 0 makes the parameter effectively an integer
    decimals: int = 3


class Parameterized:
    """Mixin holding current values for the declared ``params``."""

    params: ClassVar[dict[str, ParamSpec]] = {}

    def __init__(self) -> None:
        self.values: dict[str, float] = {
            key: spec.default for key, spec in self.params.items()
        }

    def set_param(self, name: str, value: float) -> None:
        if name not in self.values:
            raise KeyError(f"unknown parameter {name!r}")
        self.values[name] = float(value)
