import numpy.testing as npt

import ipsuite as ips


def test_mix_calculators(proj_path, traj_file):
    with ips.Project(automatic_node_names=True) as proj:
        data = ips.AddData(traj_file)
        lj1 = ips.calculators.LJSinglePoint(data=data.atoms)
        lj2 = ips.calculators.LJSinglePoint(data=data.atoms)
        lj3 = ips.calculators.LJSinglePoint(data=data.atoms)

        mix1 = ips.calculators.MixCalculator(
            data=data.atoms,
            calculators=[lj1, lj2],
            methods="mean",
        )

        mix2 = ips.calculators.MixCalculator(
            data=data.atoms,
            calculators=[lj1, lj2],
            methods="sum",
        )

        mix3 = ips.calculators.MixCalculator(
            data=data.atoms,
            calculators=[lj1, lj2, lj3],
            methods=["mean", "sum", "mean"],
        )

    proj.run()

    lj1.load()
    mix1.load()

    for a, b in zip(lj1.atoms, mix1.atoms):
        assert a.get_potential_energy() == b.get_potential_energy()
        npt.assert_almost_equal(a.get_forces(), b.get_forces())

    lj2.load()
    mix2.load()

    for a, b, c in zip(lj1.atoms, lj2.atoms, mix2.atoms):
        assert (
            a.get_potential_energy() + b.get_potential_energy()
            == c.get_potential_energy()
        )
        npt.assert_almost_equal(a.get_forces() + b.get_forces(), c.get_forces())

    lj3.load()
    mix3.load()

    for a, b, c, d in zip(lj1.atoms, lj2.atoms, lj3.atoms, mix3.atoms):

        # (a + c / 2) + b
        true_energy = a.get_potential_energy() + b.get_potential_energy()
        true_forces = a.get_forces() + b.get_forces()

        assert true_energy == d.get_potential_energy()
        npt.assert_almost_equal(true_forces, d.get_forces())


def test_mix_calculator_external(proj_path, traj_file):
    lj1 = ips.calculators.LJSinglePoint(data=None)
    lj2 = ips.calculators.LJSinglePoint(data=None)

    with ips.Project(automatic_node_names=True) as proj:
        data = ips.AddData(traj_file)
        lj3 = ips.calculators.LJSinglePoint(data=data.atoms)

        mix1 = ips.calculators.MixCalculator(
            data=data.atoms,
            calculators=[lj1, lj2],
            methods="mean",
        )

    proj.run()

    lj3.load()
    mix1.load()

    for a, b in zip(lj3.atoms, mix1.atoms):
        assert a.get_potential_energy() == b.get_potential_energy()
        npt.assert_almost_equal(a.get_forces(), b.get_forces())
