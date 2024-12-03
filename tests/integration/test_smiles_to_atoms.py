import itertools

import numpy as np

import ipsuite as ips


def test_Smiles2Atoms(proj_path):
    with ips.Project() as proj:
        ethanol = ips.Smiles2Atoms(smiles="CCO")

    proj.repro()

    assert len(ethanol.atoms) == 1
    assert ethanol.atoms[0].get_chemical_formula() == "C2H6O"


def test_SmilesToConformers(proj_path):
    with ips.Project() as proj:
        ethanol = ips.SmilesToConformers(smiles="CCO", numConfs=10)

    proj.repro()

    assert len(ethanol.frames) == 10
    assert ethanol.frames[0].get_chemical_formula() == "C2H6O"
    # iterate over all pairs of conformers and check that none are identical
    for conf1, conf2 in itertools.combinations(ethanol.frames, 2):
        assert not np.allclose(conf1.positions, conf2.positions)
