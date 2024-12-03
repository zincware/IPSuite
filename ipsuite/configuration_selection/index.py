"""Select configurations by item, e.g. slice or list of indices."""

import typing

import ase
import zntrack

from ipsuite.configuration_selection import ConfigurationSelection


class IndexSelection(ConfigurationSelection):
    """Select atoms based on getitems.

    Attributes
    ----------
    indices: list[int]|slice|
    """

    indices: list[int] | None = zntrack.params(None)
    start: int | None = zntrack.params(None)
    stop: int | None = zntrack.params(None)
    step: int | None = zntrack.params(None)

    def select_atoms(self, atoms_lst: typing.List[ase.Atoms]) -> typing.List[int]:
        """Select Atoms randomly."""
        if self.indices:
            if isinstance(self.indices, typing.Iterable):
                return self.indices
            else:
                raise ValueError("indices must be an iterable of integers")
        else:
            idx_slice = slice(self.start, self.stop, self.step)
            return list(range(len(atoms_lst)))[idx_slice]
