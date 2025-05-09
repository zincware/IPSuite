"""Abstract base classes and type hints."""
import typing as t

import zntrack
from ase.calculators.calculator import Calculator

T = t.TypeVar("T", bound=zntrack.Node, covariant=True)


class NodeWithCalculator(t.Protocol[T]):
    def get_calculator(self, **kwargs) -> Calculator: ...
