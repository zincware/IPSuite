"""Module for selecting Atoms randomly."""

import typing

import ase
import numpy as np
import zntrack

from ipsuite.configuration_selection import ConfigurationSelection


class RandomSelection(ConfigurationSelection):
    """Select atoms randomly."""

    n_configurations = zntrack.zn.params()
    seed = zntrack.zn.params(1234)

    def select_atoms(self, atoms_lst: typing.List[ase.Atoms]) -> typing.List[int]:
        """Select Atoms randomly."""
        np.random.seed(self.seed)
        return np.random.choice(
            len(atoms_lst), size=self.n_configurations, replace=False
        ).tolist()
