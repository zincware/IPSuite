import ase.io
from ase import units

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

    project.run()

    md.load()

    assert len(md.atoms) == 30
    assert md.atoms[0].cell[0, 0] == 7.22
    assert md.atoms[1].cell[0, 0] > 7.22
    assert md.atoms[1].cell[0, 0] < 10
    assert md.atoms[-1].cell[0, 0] == 10


def test_ase_md_box_ramp(proj_path, cu_box):
    ase.io.write("cu_box.xyz", cu_box)
    thermostat = ips.calculators.LangevinThermostat(
        time_step=1,
        temperature=1,
        friction=1,
    )
    rescale_box = ips.calculators.BoxOscillatingRampModifier(
        end_cell=10.0,
        cell_amplitude=2.0,
        num_oscillations=1.0,
    )
    with ips.Project() as project:
        data = ips.AddData(file="cu_box.xyz")
        model = ips.calculators.EMTSinglePoint(data=data.atoms)
        md = ips.calculators.ASEMD(
            data=data.atoms,
            model=model,
            modifier=[rescale_box],
            thermostat=thermostat,
            init_temperature=1.0,
            steps=20,
            sampling_rate=1,
            dump_rate=33,
        )

    project.run()

    md.load()

    assert len(md.atoms) == 20
    assert md.atoms[0].cell[0, 0] == 7.22
    assert md.atoms[1].cell[0, 0] > 7.22
    assert md.atoms[1].cell[0, 0] < 10
    assert (md.atoms[-1].cell[0, 0] - 10.0) < 1e-6


def test_ase_npt(proj_path, cu_box):
    ase.io.write("cu_box.xyz", cu_box)
    thermostat = ips.calculators.NPTThermostat(
        time_step=1.0,
        temperature=300,
        pressure=1.01325 * units.bar,
        ttime=25 * units.fs,
        pfactor=(75 * units.fs) ** 2,
        tetragonal_strain=True,
    )
    temperature_ramp = ips.calculators.TemperatureOscillatingRampModifier(
        end_temperature=100.0,
        temperature_amplitude=20.0,
        num_oscillations=2.0,
    )
    with ips.Project() as project:
        data = ips.AddData(file="cu_box.xyz")
        model = ips.calculators.EMTSinglePoint(data=data.atoms)
        md = ips.calculators.ASEMD(
            data=data.atoms,
            model=model,
            modifier=[temperature_ramp],
            thermostat=thermostat,
            init_temperature=1.0,
            steps=30,
            sampling_rate=1,
            dump_rate=33,
        )

    project.run()

    md.load()

    assert len(md.atoms) == 30
    assert md.atoms[0].cell[0, 0] == 7.22
    assert abs(md.atoms[1].cell[0, 0] - 7.22) > 1e-6
