"""Module for selecting Atoms randomly."""

import typing

import ase
import numpy as np
import zntrack

from ipsuite.configuration_selection import ConfigurationSelection


class SplitSelection(ConfigurationSelection):
    """Select the first n% of configurations from the dataset.

    Parameters
    ----------
    data : list[ase.Atoms]
        The atomic configurations to select from.
    split : float
        Fraction of the data to select (0.0 to 1.0).

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
    ...     selector = ips.SplitSelection(data=data.frames, split=0.1)
    >>> project.repro()
    >>> print(f"Selected {len(selector.selected_ids)} configurations with IDs: "
    ...       f"{selector.selected_ids}")
    Selected 10 configurations with IDs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """

    split: float = zntrack.params()

    def select_atoms(self, atoms_lst: typing.List[ase.Atoms]) -> typing.List[int]:
        """Select Atoms randomly."""
        return np.arange(len(atoms_lst))[: int(len(atoms_lst) * self.split)].tolist()
