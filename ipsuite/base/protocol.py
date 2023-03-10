"""Collection of protocols and complex type hints for type hinting."""
import typing

import ase


# Collection of Protocols for type hinting
class HasAtoms(typing.Protocol):
    """Protocol for objects that have an atoms attribute."""

    atoms: list[ase.Atoms]


class HasSelectedConfigurations(typing.Protocol):
    """Protocol for objects that have a selected_configurations attribute."""

    selected_configurations: typing.Dict[str, typing.List[int]]


class ProcessAtoms(typing.Protocol):
    """Protocol for objects that process atoms.

    Attributes
    ----------
    data : list[ase.Atoms]
        List of atoms to be processed.
    atoms : list[ase.Atoms]
        List of processed atoms.
    """

    data: list[ase.Atoms]
    atoms: list[ase.Atoms]


# Collection of complex type hints
ATOMS_LST = list[ase.Atoms]
UNION_ATOMS_OR_ATOMS_LST = typing.Union[ATOMS_LST, typing.List[ATOMS_LST]]

HasOrIsAtoms = typing.Union[UNION_ATOMS_OR_ATOMS_LST, HasAtoms]
