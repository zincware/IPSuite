import typing

import pytest
from ase import Atoms

from ipsuite.configuration_selection.base import (
    ConfigurationSelection,
    _flatten,
    _unflatten,
)


@pytest.fixture()
def atoms_lists() -> typing.List[typing.List[Atoms]]:
    return [
        [Atoms(atoms, positions=[(0, 0, 0), (0, 0, idx)]) for idx in range(20)]
        for atoms in ["CO", "HO"]
    ]


def test__flatten(atoms_lists):
    atoms = atoms_lists
    all_atoms, length_per_node_attr = _flatten(
        full_configurations=atoms,
        node_names=["Config_1", "Config_2"],
        exclude_configurations={"Config_1": [0], "Config_2": [1, 3]},
    )
    # assert isinstance(all_atoms, znslice.LazySequence)
    assert isinstance(all_atoms[0], Atoms)
    assert all_atoms[0] == atoms[0][1]
    assert all_atoms[19] == atoms[1][0]
    assert all_atoms[20] == atoms[1][2]
    assert all_atoms[22] == atoms[1][5]
    assert len(all_atoms) == 37
    assert isinstance(length_per_node_attr, dict)
    assert length_per_node_attr["Config_1"] == (len(atoms[0]) - 1)


def test__unflatten_exclude():
    all_ids = {"Config_1": 10, "Config_2": 10}
    selected_atoms = list(range(20))
    selection_per_group = _unflatten(
        selected_configurations=selected_atoms,
        length_per_node_attr=all_ids,
        exclude_configurations={"Config_1": [0], "Config_2": [4, 7]},
    )
    assert len(selection_per_group["Config_1"]) == 10
    assert len(selection_per_group["Config_2"]) == 10

    assert selection_per_group["Config_1"] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert selection_per_group["Config_2"] == [0, 1, 2, 3, 5, 6, 8, 9, 10, 11]


def test_post_init(atoms_lists):
    atoms = atoms_lists
    configuration_selection = ConfigurationSelection(data=None)
    excluded = {"Config_1": [3, 4], "Config_2": [10]}
    configuration_selection.data = atoms
    configuration_selection.exclude_configurations = excluded
    configuration_selection._post_init_()
    assert configuration_selection.data == atoms
    assert configuration_selection.exclude_configurations == excluded


def test_atoms(atoms_lists):
    atoms = atoms_lists
    configuration_selection = ConfigurationSelection(data=None)
    excluded = {"Config_1": [3, 4], "Config_2": [10]}
    configuration_selection.data = atoms
    configuration_selection.exclude_configurations = excluded
    selected = {"Config_1": [0, 1, 2, 5], "Config_2": list(range(10))}
    configuration_selection.selected_configurations = selected
    # assert isinstance(configuration_selection.atoms, znslice.LazySequence)
    assert isinstance(configuration_selection.atoms[0], Atoms)
    assert len(configuration_selection.atoms) == 14


def test_excluded_atoms(atoms_lists):
    atoms = atoms_lists
    configuration_selection = ConfigurationSelection(data=None)
    configuration_selection.data = atoms
    selected = {"Config_1": list(range(0, 20, 2)), "Config_2": list(range(10))}
    configuration_selection.selected_configurations = selected
    # print(configuration_selection.exclude_atoms)
    # assert isinstance(configuration_selection.excluded_atoms, znslice.LazySequence)
    assert isinstance(configuration_selection.excluded_atoms[0], Atoms)
    assert len(configuration_selection.excluded_atoms) == 20
