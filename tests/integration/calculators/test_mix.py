import numpy.testing as npt

import ipsuite as ips


def test_mix_calculators(proj_path, traj_file):
    with ips.Project(automatic_node_names=True) as proj:
        data = ips.AddData(traj_file)
        lj1 = ips.calculators.LJSinglePoint(data=data.atoms)
        lj2 = ips.calculators.LJSinglePoint(data=data.atoms)

        mean_calc = ips.calculators.MixCalculator(
            data=data.atoms,
            calculators=[lj1, lj2],
            method="mean",
        )

        sum_calc = ips.calculators.MixCalculator(
            data=data.atoms,
            calculators=[lj1, lj2],
            method="sum",
        )

    proj.run()

    lj1.load()
    mean_calc.load()

    for a, b in zip(lj1.atoms, mean_calc.atoms):
        npt.assert_almost_equal(a.get_potential_energy(), b.get_potential_energy(), decimal=2)
        npt.assert_almost_equal(a.get_forces(), b.get_forces())

    lj2.load()
    sum_calc.load()

    for a, b, c in zip(lj1.atoms, lj2.atoms, sum_calc.atoms):
        npt.assert_almost_equal(
            a.get_potential_energy() + b.get_potential_energy(), c.get_potential_energy(), decimal=2
        )
        npt.assert_almost_equal(a.get_forces() + b.get_forces(), c.get_forces())


def test_mix_calculator_external(proj_path, traj_file):
    lj1 = ips.calculators.LJSinglePoint(data=None)
    lj2 = ips.calculators.LJSinglePoint(data=None)

    with ips.Project(automatic_node_names=True) as proj:
        data = ips.AddData(traj_file)
        lj3 = ips.calculators.LJSinglePoint(data=data.atoms)

        mean_calc = ips.calculators.MixCalculator(
            data=data.atoms,
            calculators=[lj1, lj2],
            method="mean",
        )

        sum_calc = ips.calculators.MixCalculator(
            data=data.atoms,
            calculators=[lj1, lj2],
            method="sum",
        )

    proj.run()

    lj3.load()
    mean_calc.load()

    for a, b in zip(lj3.atoms, mean_calc.atoms):
        npt.assert_almost_equal(a.get_potential_energy(), b.get_potential_energy(), decimal=2)
        npt.assert_almost_equal(a.get_forces(), b.get_forces())

    lj3.load()
    sum_calc.load()

    for a, b, c in zip(lj3.atoms, lj3.atoms, sum_calc.atoms):
        npt.assert_almost_equal(
            a.get_potential_energy() + b.get_potential_energy(), c.get_potential_energy(), decimal=2
        )
        npt.assert_almost_equal(a.get_forces() + b.get_forces(), c.get_forces())
