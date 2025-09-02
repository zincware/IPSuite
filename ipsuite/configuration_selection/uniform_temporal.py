"""Module for selecting atoms uniform in time."""

import typing

import ase
import numpy as np
import zntrack

from ipsuite.configuration_selection import ConfigurationSelection


class UniformTemporalSelection(ConfigurationSelection):
    """Select configurations uniformly distributed across time.

    Parameters
    ----------
    data : list[ase.Atoms]
        The atomic configurations to select from.
    n_configurations : int
        Number of configurations to select uniformly across the trajectory.

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
    ...     selector = ips.UniformTemporalSelection(data=data.frames, n_configurations=5)
    >>> project.repro()
    >>> print(f"Selected {len(selector.selected_ids)} configurations with IDs: "
    ...       f"{selector.selected_ids}")
    Selected 5 configurations with IDs: [0, 25, 50, 74, 99]
    """

    n_configurations: int = zntrack.params()

    def select_atoms(self, atoms_lst: typing.List[ase.Atoms]) -> typing.List[int]:
        """Select Atoms uniform in time."""
        return (
            np.round(np.linspace(0, len(atoms_lst) - 1, self.n_configurations))
            .astype(int)
            .tolist()
        )
