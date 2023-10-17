import numpy as np
import pytest

from ipsuite.configuration_selection import ThresholdSelection


@pytest.mark.parametrize(
    "key, reference, dim_reduction",
    [
        ("energy_uncertainty", "energy", None),
        ("forces_uncertainty", "forces", {"max": (1, 2)}),
        ("forces_uncertainty", "forces", {"mean": (1, 2)}),
        ("forces_uncertainty", "forces", None),
    ],
)
def test_get_selected_atoms(atoms_list, key, reference, dim_reduction):
    threshold = ThresholdSelection(
        key=key,
        reference=reference,
        dim_reduction=dim_reduction,
        data=None,
        threshold=0.5,
        n_configurations=5,
        min_distance=5,
        save_fig=False,
    )

    if "forces_uncertainty" in key and dim_reduction is None:
        with pytest.raises(ValueError):
            selected_atoms = threshold.select_atoms(atoms_list)
    else:
        selected_atoms = threshold.select_atoms(atoms_list)
        test_selection = np.linspace(20, 0, 5, dtype=int).tolist()
        assert len(set(selected_atoms)) == 5
        assert isinstance(selected_atoms, list)
        assert selected_atoms == test_selection
