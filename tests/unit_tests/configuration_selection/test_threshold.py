import numpy as np
import pytest

from ipsuite.configuration_selection import ThresholdSelection


@pytest.mark.parametrize(
    "key, reference, dim_reduction, reduction_axis",
    [
        ("energy_uncertainty", "energy", None, (1, 2)),
        ("forces_uncertainty", "forces", "max", (1, 2)),
        ("forces_uncertainty", "forces", "mean", (1, 2)),
        ("forces_uncertainty", "forces", None, (1, 2)),
    ],
)
def test_get_selected_atoms(atoms_list, key, reference, dim_reduction, reduction_axis):
    threshold = ThresholdSelection(
        key=key,
        reference=reference,
        dim_reduction=dim_reduction,
        reduction_axis=reduction_axis,
        data=None,
        threshold=1.0,
        n_configurations=5,
        min_distance=5,
    )

    if "forces_uncertainty" in key and dim_reduction is None:
        with pytest.raises(ValueError):
            selected_atoms = threshold.select_atoms(atoms_list, safe_fig=False)
    else:
        selected_atoms = threshold.select_atoms(atoms_list, safe_fig=False)
        test_selection = np.linspace(20, 0, 5, dtype=int).tolist()
        assert len(set(selected_atoms)) == 5
        assert isinstance(selected_atoms, list)
        assert selected_atoms == test_selection
