"""Base Node for ConfigurationSelection."""
import typing

import ase
import znflow
import zntrack

from ipsuite import base


class ConfigurationSelection(base.ProcessAtoms):
    """Base Node for ConfigurationSelection.

    Attributes
    ----------
    data: list[Atoms]|list[list[Atoms]]|utils.types.SupportsAtoms
        the data to select from
    exclude_configurations: dict[str, list]|utils.types.SupportsSelectedConfigurations
        Atoms to exclude from the selection

    """

    exclude_configurations: typing.Union[
        typing.Dict[str, typing.List[int]], base.protocol.HasSelectedConfigurations
    ] = zntrack.zn.deps(
        None
    )  # TODO allow list of dicts that can be combined
    selected_configurations: typing.Dict[str, typing.List[int]] = zntrack.zn.outs()

    _name_ = "ConfigurationSelection"

    def run(self):
        """ZnTrack Node Run method."""
        self.selected_configurations = self.select_atoms(self.get_data())

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
    def atoms(self) -> typing.Sequence[ase.Atoms]:
        """Get a list of the selected atoms objects."""
        with znflow.disable_graph():
            atoms_lst = []
            for idx, atoms in enumerate(self.get_data()):
                if idx in self.selected_configurations:
                    atoms_lst.append(atoms)
            return atoms_lst

    @property
    def excluded_atoms(self) -> typing.Sequence[ase.Atoms]:
        """Get a list of the atoms objects that were not selected."""
        with znflow.disable_graph():
            atoms_lst = []
            for idx, atoms in enumerate(self.get_data()):
                if idx not in self.selected_configurations:
                    atoms_lst.append(atoms)
            return atoms_lst
