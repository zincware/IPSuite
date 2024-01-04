import ase.io
import numpy as np
import numpy.testing as npt
import pytest
from ase.build import molecule

import ipsuite as ips


def test_d3(proj_path):
    water = molecule("H2O")
    ase.io.write("water.xyz", water)

    with ips.Project() as proj:
        data = ips.AddData(file="water.xyz", name="water")
        d3 = ips.calculators.TorchD3(
            data=data.atoms,
            xc="pbe",
            damping="bj",
            cutoff=5,
            abc=False,
            cnthr=4,
            dtype="float32",
        )

    proj.run()

    d3.load()
    assert d3.atoms[0].get_potential_energy() == pytest.approx(-0.00978192157446211)
    npt.assert_almost_equal(
        d3.atoms[0].get_forces(),
        [
            [0.0, 0.0, 9.5722440e-05],
            [0.0, -4.0598028e-05, -4.7861220e-05],
            [0.0, 4.0598028e-05, -4.7861220e-05],
        ],
    )


def test_d3_existing_calc(proj_path):
    water = molecule("H2O")
    water.cell = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
    water.center()
    water.pbc = True
    ase.io.write("water.xyz", water)

    with ips.Project() as proj:
        data = ips.AddData(file="water.xyz", name="water")
        lj = ips.calculators.LJSinglePoint(data=data.atoms)
        d3 = ips.calculators.TorchD3(
            data=lj.atoms,
            xc="pbe",
            damping="bj",
            cutoff=5,
            abc=False,
            cnthr=4,
            dtype="float32",
        )

    proj.run()

    lj.load()
    d3.load()

    assert lj.atoms[0].get_potential_energy() == pytest.approx(1.772068860)
    assert lj.atoms[0].get_potential_energy() - d3.atoms[
        0
    ].get_potential_energy() == pytest.approx(0.00978192157)

    # assert not np.allclose(lj.atoms[0].get_forces()[0], d3.atoms[0].get_forces()[0])
    # assert not np.allclose(lj.atoms[0].get_stress()[0], d3.atoms[0].get_stress()[0])
