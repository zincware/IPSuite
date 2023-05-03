import pytest

from ipsuite.geometry.graphs import split_molecule
import numpy as np
from ase.build import molecule

def test_split_molecule():
    atoms = molecule('CH3CH2OH')
    a0 = 1
    a1 = 0
    rev_c_lists = [[0, 8, 6, 7], [1, 2, 3, 4, 5]]


    c_lists = split_molecule(a0, a1, atoms)

    assert len(c_lists) == 2
    for i in range(len(c_lists)):
        assert np.all(c_lists[i] == rev_c_lists[i])
        
    with pytest.raises(Exception):
        a0 = 2
        a1 = 0
        c_lists = split_molecule(a0, a1, atoms)