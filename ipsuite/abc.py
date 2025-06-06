from typing import Protocol, TypeVar

import ase
from ase.calculators.calculator import Calculator
from ase.optimize.optimize import Dynamics

T = TypeVar("T", covariant=True)


class NodeWithCalculator(Protocol[T]):
    """Any class with a `get_calculator` method returning an ASE Calculator."""

    def get_calculator(self, **kwargs) -> Calculator: ...


class NodeWithThermostat(Protocol[T]):
    """Any class with a `get_thermostat` method returning an ASE Dynamics."""

    @property
    def time_step(self) -> float: ...

    @property
    def temperature(self) -> float: ...

    @temperature.setter
    def temperature(self, value: float) -> None: ...

    def get_thermostat(self, atoms: ase.Atoms) -> Dynamics: ...
