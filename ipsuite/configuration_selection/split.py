"""Module for selecting Atoms randomly."""
import typing

import ase
import numpy as np
import zntrack

from ipsuite.configuration_selection import ConfigurationSelection


class SplitSelection(ConfigurationSelection):
    """Select the first n % of the data.

    Attributes
    ----------
    split : float
        The percentage of the data to select.
    """

    split = zntrack.zn.params()

    def select_atoms(self, atoms_lst: typing.List[ase.Atoms]) -> typing.List[int]:
        """Select Atoms randomly."""
        return np.arange(len(atoms_lst))[: int(len(atoms_lst) * self.split)].tolist()
