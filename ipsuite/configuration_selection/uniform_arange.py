"""Selecting atoms with a given step between them."""

import typing

import ase
import numpy as np
import zntrack

from ipsuite.configuration_selection import ConfigurationSelection


class UniformArangeSelection(ConfigurationSelection):
    """Class to perform a uniform arange action on a given atoms object list.

    Attributes
    ----------
    step: int
        setting the step, every nth (step) object will be taken
    """

    step: int = zntrack.params()

    def select_atoms(self, atoms_lst: typing.List[ase.Atoms]) -> typing.List[int]:
        """Take every nth (step) object of a given atoms list.

        Parameters
        ----------
        atoms_lst: typing.List[ase.Atoms]
            list of atoms objects to arange

        Returns
        -------
        typing.List[int]:
            list containing the taken indices
        """
        return np.arange(start=0, stop=len(atoms_lst), step=self.step, dtype=int).tolist()
