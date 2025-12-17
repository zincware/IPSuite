"""Base Node for ConfigurationSelection."""

import logging
import typing
from pathlib import Path

import ase
import matplotlib.pyplot as plt
import numpy as np
import zntrack

from ipsuite import base

log = logging.getLogger(__name__)


class ConfigurationSelection(base.IPSNode):
    """Base Node for ConfigurationSelection.

    Attributes
    ----------
    data: list[Atoms]|list[list[Atoms]]|utils.types.SupportsAtoms
        the data to select from
    exclude_configurations: dict[str, list]|utils.types.SupportsSelectedConfigurations
        Atoms to exclude from the
    exclude: list[zntrack.Node]|zntrack.Node|None
        Exclude the selected configurations from these nodes.

    """

    data: list[ase.Atoms] = zntrack.deps()
    selected_ids: list[int] = zntrack.outs(independent=True)

    img_selection: Path = zntrack.outs_path(zntrack.nwd / "selection.png")

    def get_data(self) -> list[ase.Atoms]:
        """Get the atoms data to process."""
        if self.data is not None:
            return self.data
        else:
            raise ValueError("No data given.")

    def run(self):
        """ZnTrack Node Run method."""

        log.debug(f"Selecting from {len(self.data)} configurations.")
        self.selected_ids = self.select_atoms(self.data)
        self._get_plot(self.data, self.selected_ids)

    def select_atoms(self, atoms_lst: typing.List[ase.Atoms]) -> typing.List[int]:
        """Run the selection method.

        Attributes
        ----------
        atoms_lst: List[ase.Atoms]
            List of ase Atoms objects to select configurations from.

        Returns
        -------
        List[int]:
            A list of the selected ids from 0 .. len(atoms_lst)
        """
        raise NotImplementedError

    @property
    def frames(self) -> list[ase.Atoms]:
        """Get a list of the selected atoms objects."""
        return [atoms for i, atoms in enumerate(self.data) if i in self.selected_ids]

    @property
    def excluded_frames(self) -> list[ase.Atoms]:
        """Get a list of the atoms objects that were not selected."""
        return [atoms for i, atoms in enumerate(self.data) if i not in self.selected_ids]

    def _get_plot(self, atoms_lst: typing.List[ase.Atoms], indices: typing.List[int]):
        """Plot the selected configurations."""
        # if energies are available, plot them, otherwise just plot indices over time
        fig, ax = plt.subplots()

        try:
            line_data = np.array([atoms.get_potential_energy() for atoms in atoms_lst])
            ax.set_ylabel("Energy")
        except Exception:
            line_data = np.arange(len(atoms_lst))
            ax.set_ylabel("Configuration")

        ax.plot(line_data)
        ax.scatter(indices, line_data[indices], c="r")
        ax.set_xlabel("Configuration")
        fig.savefig(self.img_selection, bbox_inches="tight")
        plt.close()


class BatchConfigurationSelection(ConfigurationSelection):
    """Base node for BatchConfigurationSelection.

    Attributes
    ----------
    data: list[ase.Atoms]
        The atoms data to process. This must be an input to the Node
    train_data: list[ase.Atoms]
        Batch active learning methods usually take into account the data
        a model was trained on. The training dataset has to be supplied
        with this argument.
    atoms: list[ase.Atoms]
        The processed atoms data. This is an output of the Node.
        It does not have to be 'field.Atoms' but can also be e.g. a 'property'.
    """

    train_data: list[ase.Atoms] = zntrack.deps()
