import numpy as np
import pytest
from ase import Atoms

import ipsuite as ips


def test_ips_BarycenterMapping(data_repo):
    """Test the BarycenterMapping class."""
    data = ips.AddData.from_rev(name="BMIM_BF4_363_15K")

    with ips.Project() as project:
        mapping = ips.geometry.BarycenterMapping(data=data.atoms)

    project.repro()

    assert len(mapping.atoms) == 30
    assert len(mapping.molecules) == 30 * 20
    assert isinstance(mapping.atoms, list)
    assert isinstance(mapping.molecules, list)

    assert isinstance(mapping.atoms[0], Atoms)
    assert isinstance(mapping.molecules[0], Atoms)

    assert len(mapping.atoms[0]) == 20

    mol_per_conf = mapping.get_molecules_per_configuration()
    assert len(mol_per_conf) == 30
    assert len(mol_per_conf[0]) == 20
    assert isinstance(mol_per_conf[0][0], Atoms)
