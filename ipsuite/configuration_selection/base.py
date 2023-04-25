"""Base Node for ConfigurationSelection."""
import logging
import typing
import uuid

import ase
import znflow
import zntrack

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
        Atoms to exclude from the
    exclude: list[zntrack.Node]|zntrack.Node|None
        Exclude the selected configurations from these nodes.

    """

    _hash = zntrack.zn.outs()
    exclude_configurations: typing.Union[
        typing.Dict[str, typing.List[int]], base.protocol.HasSelectedConfigurations
    ] = zntrack.zn.deps(None)
    exclude: typing.Union[zntrack.Node, typing.List[zntrack.Node]] = zntrack.zn.deps(None)
    selected_configurations: typing.Dict[str, typing.List[int]] = zntrack.zn.outs()

    _name_ = "ConfigurationSelection"

    def _post_init_(self):
        if self.data is not None and not isinstance(self.data, dict):
            try:
                self.data = znflow.combine(
                    self.data, attribute="atoms", return_dict_attr="name"
                )
            except TypeError:
                self.data = znflow.combine(self.data, attribute="atoms")

    def run(self):
        """ZnTrack Node Run method."""
        self._hash = str(uuid.uuid4())
        if self.exclude is not None:
            if self.exclude_configurations is None:
                self.exclude_configurations = {}
            if not isinstance(self.exclude, list):
                self.exclude = [self.exclude]
            for exclude in self.exclude:
                for key in exclude.selected_configurations:
                    if key in self.exclude_configurations:
                        self.exclude_configurations[key].extend(
                            exclude.selected_configurations[key]
                        )
                    else:
                        self.exclude_configurations[key] = (
                            exclude.selected_configurations[key]
                        )

        exclude = combine.ExcludeIds(self.get_data(), self.exclude_configurations)
        data = exclude.get_clean_data(flatten=True)

        log.debug(f"Selecting from {len(data)} configurations.")

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
            if isinstance(data, list) and isinstance(self.selected_configurations, list):
                for idx, atoms in enumerate(data):
                    if idx not in self.selected_configurations:
                        results.append(atoms)
            elif isinstance(data, dict) and isinstance(
                self.selected_configurations, dict
            ):
                for key, atoms_lst in data.items():
                    if key not in self.selected_configurations:
                        results.extend(atoms_lst)
                    else:
                        for idx, atoms in enumerate(atoms_lst):
                            if idx not in self.selected_configurations[key]:
                                results.append(atoms)
            else:
                raise ValueError(f"Data must be a list or dict, not {type(data)}")
            return results
