from typing import List

import ase
import pytest

from ipsuite import AddData


@pytest.fixture()
def atoms_list() -> List[ase.Atoms]:
    return [ase.Atoms("CO", positions=[(0, 0, 0), (0, 0, idx)]) for idx in range(10)]


