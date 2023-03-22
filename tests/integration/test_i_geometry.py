import pytest
import numpy as np
from ase import Atoms
import pytest
import ipsuite as ips


@pytest.mark.parametrize("eager", [True, False])
def test_ips_BarycenterMapping(data_repo, eager):
    """Test the BarycenterMapping class."""
    data = ips.AddData.from_rev(name="BMIM_BF4_363_15K")

    with ips.Project() as project:
        mapping = ips.geometry.BarycenterMapping(data=data.atoms)

    project.run(eager=eager)
    if not eager:
        mapping.load()

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
