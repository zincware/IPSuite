"""Base Node for ConfigurationSelection."""
import typing

import ase
import numpy as np
import znflow
import znslice
import zntrack

from ipsuite import base, utils


def _flatten(
    full_configurations: base.protocol.UNION_ATOMS_OR_ATOMS_LST,
    node_names: typing.List[str],
    exclude_configurations: dict = None,
) -> typing.Tuple[typing.List[ase.Atoms], dict]:
    """Make a flattened list of ase Atoms objects and exclude by id.

    Parameters
    ----------
    full_configurations: list[Atoms]|list[list[Atoms]]
        The data to flatten
    node_names: list[str]
        A list of the node names gatherd from NodeAttribute in the same order as the
         elements in full_configurations
    exclude_configurations: dict[str, list[int]]
        A dictionary of names from NodeAttribute containing indices to exclude.

    Returns
    -------
    all_atoms: list[Atoms],
        List of flattened out ase Atoms objects from the given configuration
    length_per_node_attr: dict[str, int],
        A dictionary of the groups and the respective length per group

    """
    if exclude_configurations is None:
        exclude_configurations = {}

    all_atoms = znslice.LazySequence.from_obj([])
    length_per_node_attr = {}

    for atoms_lst, node_name in zip(full_configurations, node_names, strict=True):
        indices = [
            index
            for index in range(len(atoms_lst))
            if index not in exclude_configurations.get(node_name, [])
        ]
        all_atoms += znslice.LazySequence.from_obj(atoms_lst, indices=indices)

        length_per_node_attr[node_name] = len(indices)

    return all_atoms, length_per_node_attr


def _unflatten(
    selected_configurations, length_per_node_attr, exclude_configurations
) -> typing.Dict[str, typing.List[int]]:
    """Convert a flat list of ids to a dict of {group: ids}.

    the ids per group are 0based.

    Parameters
    ----------
    selected_configurations: list[int]
        selected ids from all data source, 0based
    length_per_node_attr: dict[str, int]
        ...
    exclude_configurations: dict[str, list[int]]
        List of the excluded configurations per Group that must be added back
        to the list to have the correct exclusion

    Returns
    -------
    selection_per_group: dict
        The selected configurations per group

    """
    if exclude_configurations is None:
        exclude_configurations = {}

    selected_configurations = np.array(selected_configurations)

    selection_per_group = {}

    for dataset_name, dataset_size in length_per_node_attr.items():
        # iterate over all datasets

        _selection = selected_configurations[selected_configurations < dataset_size]

        selected_configurations = selected_configurations[
            selected_configurations >= dataset_size
        ]
        # make ids 0based for the next dataset
        selected_configurations -= dataset_size

        # Shift ids by exclude_configurations
        for excluded_id in exclude_configurations.get(dataset_name, []):
            _selection[_selection >= excluded_id] += 1
        selection_per_group[dataset_name] = np.sort(_selection).tolist()

    return selection_per_group


def _merge_dicts(dicts: typing.List[typing.Dict[str, list]]) -> typing.Dict[str, list]:
    """Merge a list of dictionaries into a single one.

    Each dictionary contain a list of ids per key which are combined into a single
    dictionary with all keys and the lists added together in no particular order.
    """
    return_dicts = {}
    for mydict in dicts:
        for key, value in mydict.items():
            return_dicts[key] = return_dicts.get(key, []) + value
    return return_dicts


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

    @znflow.disable_graph()
    def _post_init_(self):
        self.data = utils.helpers.get_deps_if_node(self.data, "atoms")
        self.exclude_configurations = utils.helpers.get_deps_if_node(
            self.exclude_configurations, "selected_configurations"
        )

    def run(self):
        """ZnTrack Node Run method."""
        # TODO either pass a dict {node.name: data} or a list
        # node_attrs = get_origin(self, "data")
        # # this is bad practice, because you can not use the Node
        # #   without DVC at this point
        # if isinstance(node_attrs, list):
        #     node_names = [x.name for x in node_attrs]
        # else:
        #     node_names = [node_attrs.name]

        if isinstance(self.data[0], ase.Atoms):
            self.data = [self.data]

        node_names = list(range(len(self.data)))

        if isinstance(self.exclude_configurations, (list, tuple)):
            self.exclude_configurations = _merge_dicts(self.exclude_configurations)

        atoms_lst, length_per_node_attr = _flatten(
            full_configurations=self.data,
            node_names=node_names,
            exclude_configurations=self.exclude_configurations,
        )
        selected_configurations = self.select_atoms(atoms_lst)

        self.selected_configurations = _unflatten(
            selected_configurations, length_per_node_attr, self.exclude_configurations
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
            if isinstance(self.data[0], ase.Atoms):
                self.data = [self.data]

            selected_data = znslice.LazySequence.from_obj([])

            for selected_confs, data in zip(
                self.selected_configurations.values(), self.data, strict=True
            ):
                selected_data += znslice.LazySequence.from_obj(
                    data, indices=selected_confs
                )
            return selected_data

    @property
    def excluded_atoms(self) -> typing.Sequence[ase.Atoms]:
        """Get a list of the atoms objects that were not selected."""
        with znflow.disable_graph():
            if isinstance(self.data[0], ase.Atoms):
                # this will read the first entry, therefore, tqdm starts usually at len - 1
                self.data = [self.data]

            selected_data = znslice.LazySequence.from_obj([])
            for selected_confs, data in zip(
                self.selected_configurations.values(), self.data, strict=True
            ):
                excluded_indices = [
                    x for x in range(len(data)) if x not in selected_confs
                ]
                selected_data += znslice.LazySequence.from_obj(
                    data, indices=excluded_indices
                )
            return selected_data
