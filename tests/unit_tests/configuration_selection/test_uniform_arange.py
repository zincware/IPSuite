import numpy as np

from ipsuite.configuration_selection.uniform_arange import UniformArangeSelection


def test_get_selected_atoms(atoms_list):
    uniform_arange = UniformArangeSelection(step=5, data=None)
    selected_atoms = uniform_arange.select_atoms(atoms_list)
    test_selection = np.arange(
        0, len(atoms_list), uniform_arange.step, dtype=int
    ).tolist()
    assert len(set(selected_atoms)) == 5
    assert isinstance(selected_atoms, list)
    assert selected_atoms == test_selection
