import numpy.testing as npt

import ipsuite as ips


def test_mix_calculators(proj_path, traj_file):
    lj1 = ips.LJSinglePoint()
    lj2 = ips.LJSinglePoint()
    with ips.Project() as proj:
        data = ips.AddData(file=traj_file)

        mean_calc = ips.MixCalculator(
            calculators=[lj1, lj2],
            method="mean",
        )

        sum_calc = ips.MixCalculator(
            calculators=[lj1, lj2],
            method="sum",
        )

        lj1_data = ips.Prediction(data=data.frames, model=lj1)
        lj2_data = ips.Prediction(data=data.frames, model=lj2)
        mean_calc_data = ips.Prediction(data=data.frames, model=mean_calc)
        sum_calc_data = ips.Prediction(data=data.frames, model=sum_calc)

    proj.repro()

    for a, b in zip(lj1_data.frames, mean_calc_data.frames):
        npt.assert_almost_equal(
            a.get_potential_energy(), b.get_potential_energy(), decimal=2
        )
        npt.assert_almost_equal(a.get_forces(), b.get_forces())

    for a, b, c in zip(lj1_data.frames, lj2_data.frames, sum_calc_data.frames):
        npt.assert_almost_equal(
            a.get_potential_energy() + b.get_potential_energy(),
            c.get_potential_energy(),
            decimal=2,
        )
        npt.assert_almost_equal(a.get_forces() + b.get_forces(), c.get_forces())
