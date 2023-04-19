import ipsuite as ips


def test_ase_md(proj_path, traj_file):
    checker = ips.analysis.TemperatureCheck()
    thermostat = ips.calculators.LagevinThermostat(
        time_step=1,
        temperature=1,
        friction=1,
    )
    with ips.Project() as project:
        data = ips.AddData(file=traj_file)
        model = ips.models.GAP(data=data.atoms)
        md = ips.calculators.ASEMD(
            data=data.atoms,
            calculator=model.calc,
            checker_list=[checker],
            thermostat=thermostat,
            init_temperature=1.0,
            steps=100,
            sampling_rate=1,
            dump_rate=33,
        )

    project.run(environment={"OPENBLAS_NUM_THREADS": "1"})

    md.load()

    assert len(md.atoms) == 100
