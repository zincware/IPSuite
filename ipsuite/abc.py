"""Abstract base classes and type hints."""

import typing as t

from ase.calculators.calculator import Calculator

T = t.TypeVar("T", covariant=True)


class NodeWithCalculator(t.Protocol[T]):
    """Any @dataclass, including zntrack.Node that provides a calculator."""
    def get_calculator(self, **kwargs) -> Calculator: ...
