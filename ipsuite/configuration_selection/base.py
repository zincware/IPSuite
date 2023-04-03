"""Base Node for ConfigurationSelection."""
import typing

import ase
import znflow
import zntrack
import logging
from ipsuite import base
from ipsuite.utils import combine

log = logging.getLogger(__name__)


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

        exclude = combine.ExcludeIds(self.get_data(), self.exclude_configurations)
        data = exclude.get_clean_data(flatten=True)

        assert isinstance(data, list), f"data must be a list, not {type(data)}"
        log.critical(f"Selecting from {len(data)} configurations.")

        selected_configurations = self.select_atoms(data)

        self.selected_configurations = exclude.get_original_ids(
            selected_configurations, per_key=True
        )

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
            results = []
            data = self.get_data()
            if isinstance(data, list):
                for idx, atoms in enumerate(self.get_data()):
                    if idx in self.selected_configurations:
                        results.append(atoms)
            elif isinstance(data, dict):
                for key, atoms_lst in data.items():
                    if key in self.selected_configurations:
                        for idx, atoms in enumerate(atoms_lst):
                            if idx in self.selected_configurations[key]:
                                results.append(atoms)
            else:
                raise ValueError(f"Data must be a list or dict, not {type(data)}")
            return results

    @property
    def excluded_atoms(self) -> typing.Sequence[ase.Atoms]:
        """Get a list of the atoms objects that were not selected."""
        with znflow.disable_graph():
            results = []
            data = self.get_data()
            if isinstance(data, list):
                for idx, atoms in enumerate(self.get_data()):
                    if idx not in self.selected_configurations:
                        results.append(atoms)
            elif isinstance(data, dict):
                for key, atoms_lst in data.items():
                    if key not in self.selected_configurations:
                        for idx, atoms in enumerate(atoms_lst):
                            if idx not in self.selected_configurations[key]:
                                results.append(atoms)
            else:
                raise ValueError(f"Data must be a list or dict, not {type(data)}")
            return results
