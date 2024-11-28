import typing

import ase
import numpy as np
import pytest
from ase import Atoms

import ipsuite as ips


def test_ips_BarycenterMapping(data_repo):
    """Test the BarycenterMapping class."""
    data = ips.AddData.from_rev(
        name="BMIM_BF4_363_15K", remote="https://github.com/IPSProjects/ips-examples"
    )

    mapping = ips.geometry.BarycenterMapping()

    frames = []
    all_molecules = []
    for atoms in data.frames:
        cg_atoms, molecules = mapping.forward_mapping(atoms)
        frames.append(cg_atoms)
        all_molecules.extend(molecules)

    assert len(frames) == 30
    assert len(all_molecules) == 30 * 20
