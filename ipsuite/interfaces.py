import pathlib
from typing import Dict, List, Protocol, TypeVar, Union

import ase
from ase.calculators.calculator import Calculator
from ase.optimize.optimize import Dynamics

T = TypeVar("T", covariant=True)


class NodeWithCalculator(Protocol[T]):
    """Any class with a `get_calculator` method returning an ASE Calculator."""

    def get_calculator(
        self, *, directory: str | pathlib.Path | None = None, **kwargs
    ) -> Calculator: ...


class NodeWithThermostat(Protocol[T]):
    """Any class with a `get_thermostat` method returning an ASE Dynamics."""

    @property
    def time_step(self) -> float: ...

    @property
    def temperature(self) -> float: ...

    @temperature.setter
    def temperature(self, value: float) -> None: ...

    def get_thermostat(self, atoms: ase.Atoms) -> Dynamics: ...


class HasAtoms(Protocol):
    """Protocol for objects that have an atoms attribute."""

    frames: list[ase.Atoms]


class HasSelectedConfigurations(Protocol):
    """Protocol for objects that have a selected_configurations attribute."""

    selected_configurations: Dict[str, List[int]]


class ProcessAtoms(Protocol):
    """Protocol for objects that process atoms.

    Parameters
    ----------
    data : list[ase.Atoms]
        List of atoms to process.

    Attributes
    ----------
    frames : list[ase.Atoms]
        List of processed atoms.
    """

    data: list[ase.Atoms]
    frames: list[ase.Atoms]


# Collection of complex type hints
ATOMS_LST = list[ase.Atoms]
UNION_ATOMS_OR_ATOMS_LST = Union[ATOMS_LST, List[ATOMS_LST]]

HasOrIsAtoms = Union[UNION_ATOMS_OR_ATOMS_LST, HasAtoms]

__all__ = [
    "NodeWithCalculator",
    "NodeWithThermostat",
    "HasAtoms",
    "HasSelectedConfigurations",
    "ProcessAtoms",
    "ATOMS_LST",
    "UNION_ATOMS_OR_ATOMS_LST",
    "HasOrIsAtoms",
]


def interfaces() -> dict[str, list[str]]:
    """Return a dictionary of available interfaces."""
    return {"ipsuite.interfaces": __all__}
