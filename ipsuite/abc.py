from typing import Protocol, TypeVar
from pathlib import Path
from ase.calculators.calculator import Calculator

T = TypeVar("T", covariant=True)

class NodeWithCalculator(Protocol[T]):
    """Any class with a `get_calculator` method returning an ASE Calculator."""

    def get_calculator(
        self, **kwargs
    ) -> Calculator: ...
