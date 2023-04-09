import numpy as np

from ipsuite.analysis.model.math import force_decomposition
from ipsuite.geometry import BarycenterMapping


def test_force_decomposition(atoms_with_composed_forces):
    atoms, ft, fr, fv = atoms_with_composed_forces

    mapping = BarycenterMapping(data=None)
    atom_trans_forces, atom_rot_forces, atom_vib_forces = force_decomposition(
        atoms, mapping
    )

    assert np.allclose(ft, atom_trans_forces)
    assert np.allclose(fr, atom_rot_forces)
    assert np.allclose(fv, atom_vib_forces)
