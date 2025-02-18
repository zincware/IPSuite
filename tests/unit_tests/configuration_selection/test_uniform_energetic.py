import numpy as np

from ipsuite.configuration_selection.uniform_energetic import \
    UniformEnergeticSelection


def test_get_selected_atoms(atoms_list):
    uniform_energetic = UniformEnergeticSelection(n_configurations=5, data=None)
    selected_atoms = uniform_energetic.select_atoms(atoms_list)
    test_selection = np.linspace(0, 20, 5, dtype=int).tolist()
    assert len(set(selected_atoms)) == 5
    assert isinstance(selected_atoms, list)
    assert selected_atoms == test_selection
