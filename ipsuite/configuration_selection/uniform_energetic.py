"""Module for selecting atoms uniformly in energy space."""

import logging
import typing

import ase
import numpy as np
import zntrack

from ipsuite.configuration_selection import ConfigurationSelection

log = logging.getLogger(__name__)


class UniformEnergeticSelection(ConfigurationSelection):
    """A class to perform data selection based on uniform global energy selection."""

    n_configurations = zntrack.params()

    def select_atoms(self, atoms_lst: typing.List[ase.Atoms]) -> typing.List[int]:
        """Select Atoms uniform in energy space."""
        log.warning(f"Running search for {self.n_configurations} to max {len(atoms_lst)}")

        data = np.array([x.get_potential_energy() for x in atoms_lst])

        sorted_data = data[np.argsort(data)]
        indices = []

        for anchor_size in range(self.n_configurations, len(sorted_data)):
            temporary_anchor = np.linspace(
                start=sorted_data[0],
                stop=sorted_data[-1],
                num=anchor_size,
            )

            if anchor_size > self.n_configurations and anchor_size % 100 == 0:
                log.debug("Extending Search range!")

            indices = np.searchsorted(sorted_data, temporary_anchor, side="left")
            indices = np.unique(indices)

            if len(indices) >= self.n_configurations:
                break

        return np.arange(len(data))[np.argsort(data)][indices].tolist()
