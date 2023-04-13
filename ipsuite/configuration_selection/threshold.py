"""Selecting atoms with a given step inbetween."""
import typing

import ase
import numpy as np
import zntrack

from ipsuite.configuration_selection import ConfigurationSelection


class ThresholdSelection(ConfigurationSelection):
    """Select atoms based on a given threshold.

    Select atoms above a given threshold or the n_configurations with the
    highest / lowest value. Typically useful for uncertainty based selection.

    Attributes
    ----------
    key: str
        the key in 'calc.results' to select from
    threshold: float, optional
        All values above (or below if negative) this threshold will be selected.
        If n_configurations is given, this threshold will be prioritized.
    n_configurations: int, optional
        number of configurations to select.
        This will only be used if threshold is not given.
    """

    key = zntrack.zn.params("energy_uncertainty")
    threshold = zntrack.zn.params(None)
    n_configurations = zntrack.zn.params(None)

    def _post_init_(self):
        if self.threshold is None and self.n_configurations is None:
            raise ValueError("Either 'threshold' or 'n_configurations' must not be None.")

        return super()._post_init_()

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
        values = np.array([atoms.calc.results[self.key] for atoms in atoms_lst])
        if self.threshold is not None:
            if self.threshold < 0:
                indices = np.where(values < self.threshold)[0]
            else:
                indices = np.where(values > self.threshold)[0]
        else:
            if np.mean(values) > 0:
                indices = np.argsort(values)[::-1][: self.n_configurations]
            else:
                indices = np.argsort(values)[: self.n_configurations]

        return indices.tolist()
