"""Module for selecting atoms uniform in time."""

import typing

import ase
import numpy as np
import zntrack

from ipsuite.configuration_selection import ConfigurationSelection


class UniformTemporalSelection(ConfigurationSelection):
    """Select atoms uniform in time."""

    n_configurations: int = zntrack.params()

    def select_atoms(self, atoms_lst: typing.List[ase.Atoms]) -> typing.List[int]:
        """Select Atoms uniform in time."""
        return (
            np.round(np.linspace(0, len(atoms_lst) - 1, self.n_configurations))
            .astype(int)
            .tolist()
        )
