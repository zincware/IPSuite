from typing import List

import ase
import pytest

from ipsuite import AddData


@pytest.fixture()
def atoms_list() -> List[ase.Atoms]:
    return [ase.Atoms("CO", positions=[(0, 0, 0), (0, 0, idx)]) for idx in range(10)]


def test_iter(atoms_list):
    add_data = AddData(file="")
    add_data.atoms = atoms_list

    assert list(add_data) == atoms_list

    assert len(add_data) == 10

    assert add_data[1] == atoms_list[1]
    assert add_data[[1, 3]] == [atoms_list[1], atoms_list[3]]
