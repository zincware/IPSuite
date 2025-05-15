"""Abstract base classes and type hints."""

import typing as t
from pathlib import Path

from ase.calculators.calculator import Calculator

T = t.TypeVar("T", covariant=True)


class NodeWithCalculator(t.Protocol[T]):
    """Any @dataclass, including zntrack.Node that provides a calculator."""

    def get_calculator(
        self, directory: str | Path | None = None, **kwargs
    ) -> Calculator: ...
