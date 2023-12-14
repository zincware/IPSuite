import numpy as np
import pytest

from ipsuite.configuration_selection import PropertyFilter


@pytest.mark.parametrize(
    "key, cutoff_type, direction, cutoffs",
    [
        ("forces", "direct", "both", [7, 13]),
        ("forces", "direct", "both", None),
        ("forces", "around_mean", "both", None),
    ],
)
def test_get_selected_atoms(atoms_list, key, cutoff_type, direction, cutoffs):
    for idx, atoms in enumerate(atoms_list):
        atoms.calc.results[key] = np.array([[idx, 0, 0], [0, 0, 0]])

    filter = PropertyFilter(
        key=key,
        cutoff_type=cutoff_type,
        direction=direction,
        data=None,
        cutoffs=cutoffs,
        threshold=0.4,
    )

    if "direct" in cutoff_type and cutoffs is None:
        with pytest.raises(ValueError):
            selected_atoms = filter.select_atoms(atoms_list)
    else:
        test_selection = [8, 9, 10, 11, 12]
        selected_atoms = filter.select_atoms(atoms_list)
        assert isinstance(selected_atoms, list)
        assert len(set(selected_atoms)) == 5
        assert selected_atoms == test_selection
