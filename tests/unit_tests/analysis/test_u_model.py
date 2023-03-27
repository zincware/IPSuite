import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from ipsuite.analysis.model import force_decomposition
from ipsuite.geometry import BarycenterMapping


def test_force_decomposition():
    atoms = Atoms(
        "OH2",
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
    )
    ft = np.array(
        [
            [0.0, 0.0, 1.0 * 15.999],
            [0.0, 0.0, 1.008],
            [0.0, 0.0, 1.008],
        ]
    )
    fr = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.008],
            [0.0, 0.0, -1.008],
        ]
    )
    fv = np.array(
        [
            [1.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    atoms.calc = SinglePointCalculator(atoms, forces=ft + fr + fv)

    mapping = BarycenterMapping(data=None)
    atom_trans_forces, atom_rot_forces, atom_vib_forces = force_decomposition(
        atoms, mapping
    )

    assert np.allclose(ft, atom_trans_forces)
    assert np.allclose(fr, atom_rot_forces)
    assert np.allclose(fv, atom_vib_forces)
