import itertools

import numpy as np

import ipsuite as ips


def test_SmilesToAtoms(proj_path):
    with ips.Project() as proj:
        ethanol = ips.configuration_generation.SmilesToAtoms(smiles="CCO")

    proj.run()
    proj.load()

    assert len(ethanol.atoms) == 1
    assert ethanol.atoms[0].get_chemical_formula() == "C2H6O"


def test_SmilesToConformers(proj_path):
    with ips.Project() as proj:
        ethanol = ips.configuration_generation.SmilesToConformers(
            smiles="CCO", numConfs=10
        )

    proj.run()
    proj.load()

    assert len(ethanol.atoms) == 10
    assert ethanol.atoms[0].get_chemical_formula() == "C2H6O"
    # iterate over all pairs of conformers and check that none are identical
    for conf1, conf2 in itertools.combinations(ethanol.atoms, 2):
        assert not np.allclose(conf1.positions, conf2.positions)
