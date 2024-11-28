import numpy as np
from ase import Atoms

import ipsuite as ips


def test_unwrapping():
    """
    Tests a relatively complex scenario in which simultaneously an atom wanders
    out of the cell while another molecule moves around the center.
    Previously there was a bug in which the unwrapping mistakenly used the
    closest atom to the center to determine the unwrapping direction as opposed to
    the closest atom from the molecule under consideration.
    """
    pos = [
        [0.1, 0.0, 0.0],  # corner molecule
        [0.4, 0.0, 0.0],
        [2.25, 2.25, 2.25],  # center molecule
        [2.0, 2.25, 2.25],
    ]

    cell = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
    atoms = Atoms("H4", positions=pos, cell=cell, pbc=True)
    atoms.wrap()
    mapping = ips.geometry.BarycenterMapping()

    cg_atoms_0, _ = mapping.forward_mapping(atoms)

    displacement = np.array([0.2, 0.0, 0.0])
    # move 1 atom out of the cell
    atoms.positions[0] -= displacement
    # change overall closest atom to the center
    atoms.positions[2:] += np.array([0.5, 0.5, 0.5])
    atoms.wrap()

    cg_atoms_1, _ = mapping.forward_mapping(atoms)

    pos0 = cg_atoms_0.positions[0]  # compare only reference molecule
    pos1 = cg_atoms_1.positions[0]

    assert np.allclose(pos0 - pos1, displacement / 2)
