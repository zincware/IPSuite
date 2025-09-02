"""Selecting atoms with a given step between them."""

import typing

import ase
import numpy as np
import zntrack

from ipsuite.configuration_selection import ConfigurationSelection


class UniformArangeSelection(ConfigurationSelection):
    """Select configurations with uniform spacing using a step size.

    Parameters
    ----------
    data : list[ase.Atoms]
        The atomic configurations to select from.
    step : int
        Step size for selection. Every nth configuration will be selected.

    Attributes
    ----------
    selected_ids : list[int]
        Indices of selected configurations.
    frames : list[ase.Atoms]
        The selected atomic configurations.
    excluded_frames : list[ase.Atoms]
        The atomic configurations that were not selected.

    Examples
    --------
    >>> with project:
    ...     data = ips.AddData(file="ethanol.xyz")  # contains 100 frames
    ...     selector = ips.UniformArangeSelection(data=data.frames, step=10)
    >>> project.repro()
    >>> print(f"Selected {len(selector.selected_ids)} configurations with IDs: "
    ...       f"{selector.selected_ids}")
    Selected 10 configurations with IDs: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
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
