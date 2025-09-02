"""Module for selecting Atoms randomly."""

import typing

import ase
import numpy as np
import zntrack

from ipsuite.configuration_selection import ConfigurationSelection


class RandomSelection(ConfigurationSelection):
    """Select configurations randomly without replacement.

    Parameters
    ----------
    data : list[ase.Atoms]
        The atomic configurations to select from.
    n_configurations : int
        Number of configurations to select.
    seed : int, default=1234
        Random seed for reproducible selection.

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
    ...     data = ips.AddData(file="ethanol.xyz")
    ...     selector = ips.RandomSelection(data=data.frames, n_configurations=10, seed=42)
    >>> project.repro()
    >>> print(f"Selected {len(selector.selected_ids)} configurations with IDs: "
    ...       f"{selector.selected_ids}")
    Selected 10 configurations with IDs: [83, 53, 70, 45, 44, 39, 22, 80, 10, 0]
    """

    n_configurations: int = zntrack.params()
    seed: int = zntrack.params(1234)

    def select_atoms(self, atoms_lst: typing.List[ase.Atoms]) -> typing.List[int]:
        """Select Atoms randomly."""
        np.random.seed(self.seed)
        return np.random.choice(
            len(atoms_lst), size=self.n_configurations, replace=False
        ).tolist()
