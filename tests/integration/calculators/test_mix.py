import ipsuite as ips
import numpy.testing as npt

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
    
    proj.run()

    lj1.load()
    mix1.load()

    for a, b in zip(lj1.atoms, mix1.atoms):
        assert a.get_potential_energy() == b.get_potential_energy()
        npt.assert_almost_equal(a.get_forces(), b.get_forces())

    lj2.load()
    mix2.load()

    for a, b, c in zip(lj1.atoms, lj2.atoms, mix2.atoms):
        assert a.get_potential_energy() + b.get_potential_energy() == c.get_potential_energy()
        npt.assert_almost_equal(a.get_forces() + b.get_forces(), c.get_forces())
