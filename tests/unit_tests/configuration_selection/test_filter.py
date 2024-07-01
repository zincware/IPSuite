import numpy as np
import pytest

from ipsuite.configuration_selection import PropertyFilter
from ipsuite.configuration_selection.filter import REDUCTIONS


@pytest.mark.parametrize(
    "reference, dim_reduction, reduction_axis",
    [
        ("energy", None, (1, 2)),
        ("forces", "max", (1, 2)),
        ("forces_uncertainty", "mean", (1, 2)),
        ("forces_uncertainty", None, (1, 2)),
    ],
)
@pytest.mark.parametrize(
    "direction",
    [
        "above",
        "below",
        "both",
    ],
)
def test_get_selected_atoms(
    atoms_list, reference, dim_reduction, reduction_axis, direction
):
    values = np.array([atoms.calc.results[reference] for atoms in atoms_list])
    if dim_reduction is not None:
        reduction_fn = REDUCTIONS[dim_reduction]
        values = reduction_fn(values, reduction_axis)

    mean = np.mean(values)
    std = np.std(values)
    upper_limit = mean + 0.5 * std
    lower_limit = mean - 0.5 * std

    filter = PropertyFilter(
        reference=reference,
        dim_reduction=dim_reduction,
        reduction_axis=reduction_axis,
        data=None,
        cutoffs=[lower_limit, upper_limit],
        n_configurations=3,
        min_distance=1,
        direction=direction,
    )

    if reference in ["forces", "forces_uncertainty"] and dim_reduction is None:
        with pytest.raises(ValueError):
            selected_atoms = filter.select_atoms(atoms_list)
    else:
        selected_atoms = filter.select_atoms(atoms_list)

        assert len(set(selected_atoms)) == 3
        assert isinstance(selected_atoms, list)

        if direction == "above":
            assert np.argmax(values) in selected_atoms

        elif direction == "below":
            assert np.argmin(values) in selected_atoms

        else:
            assert np.argmin(values) in selected_atoms
            assert np.argmax(values) in selected_atoms
