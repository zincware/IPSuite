"""Select configurations by item, e.g. slice or list of indices."""

"""Module for selecting Atoms randomly."""
import typing

import ase
import numpy as np
import zntrack

from ipsuite.configuration_selection import ConfigurationSelection


class IndexSelection(ConfigurationSelection):
    """Select atoms based on getitems.

    Attributes
    ----------
    indices: list[int]|slice|
    """

    indices = zntrack.zn.params()

    def select_atoms(self, atoms_lst: typing.List[ase.Atoms]) -> typing.List[int]:
        """Select Atoms randomly."""
        if isinstance(self.indices, list):
            return self.indices
        return list(range(len(atoms_lst)))[self.indices]
