import ipsuite as ips


def test_ase_md(proj_path, traj_file):
    checker = ips.calculators.TemperatureCheck()
    thermostat = ips.calculators.LagevinThermostat(
        time_step=1,
        temperature=1,
        friction=1,
    )
    with ips.Project() as project:
        data = ips.AddData(file=traj_file)
        model = ips.models.GAP(data=data.atoms, OPENBLAS_NUM_THREADS="1")
        md = ips.calculators.ASEMD(
            data=data.atoms,
            calculator=model.calc,
            checker_list=[checker],
            thermostat=thermostat,
            init_temperature=1,
            steps=100,
            sampling_rate=1,
            dump_rate=33,
        )

    project.run()

    md.load()

    assert len(md.atoms) == 100
