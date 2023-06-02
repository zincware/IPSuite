import ase.io

import ipsuite as ips


def test_ase_md(proj_path, cu_box):
    ase.io.write("cu_box.xyz", cu_box)
    checker = ips.analysis.TemperatureCheck()
    thermostat = ips.calculators.LangevinThermostat(
        time_step=1,
        temperature=1,
        friction=1,
    )
    rescale_box = ips.calculators.RescaleBoxModifier(cell=10)
    temperature_ramp = ips.calculators.TemperatureRampModifier(temperature=300)
    with ips.Project() as project:
        data = ips.AddData(file="cu_box.xyz")
        model = ips.calculators.EMTSinglePoint(data=data.atoms)
        md = ips.calculators.ASEMD(
            data=data.atoms,
            model=model,
            checker_list=[checker],
            modifier=[rescale_box, temperature_ramp],
            thermostat=thermostat,
            init_temperature=1.0,
            steps=30,
            sampling_rate=1,
            dump_rate=33,
        )

    project.run(environment={"OPENBLAS_NUM_THREADS": "1"})

    md.load()

    assert len(md.atoms) == 30
    assert md.atoms[0].cell[0, 0] == 7.22
    assert md.atoms[1].cell[0, 0] > 7.22
    assert md.atoms[1].cell[0, 0] < 10
    assert md.atoms[-1].cell[0, 0] == 10
