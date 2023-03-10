from ipsuite.configuration_selection.random import RandomSelection


def test_get_selected_atoms(atoms_list):
    random = RandomSelection(n_configurations=10, data=None)
    random_selection = random.select_atoms(atoms_list)
    assert len(set(random_selection)) == 10
    assert isinstance(random_selection, list)
    assert isinstance(random_selection[0], int)
