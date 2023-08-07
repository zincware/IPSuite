import ase.io
import numpy as np
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


def test_ase_md_fixed_sphere(proj_path, cu_box):
    ase.io.write("cu_box.xyz", cu_box)
    thermostat = ips.calculators.LangevinThermostat(
        time_step=1,
        temperature=1,
        friction=1,
    )

    constraint = ips.calculators.FixedSphereConstraint(
        atom_id=0,
        radius=2.6,
    )

    with ips.Project() as project:
        data = ips.AddData(file="cu_box.xyz")
        model = ips.calculators.EMTSinglePoint(data=data.atoms)
        md = ips.calculators.ASEMD(
            data=data.atoms,
            model=model,
            thermostat=thermostat,
            init_temperature=1.0,
            steps=30,
            sampling_rate=1,
            dump_rate=33,
            constraint_list=[constraint],
        )

    project.run()

    md.load()

    assert np.sum(md.atoms[0][0].position - md.atoms[-1][0].position) < 1e-6
    # neighbor atoms should not move
    assert np.sum(md.atoms[0][1].position - md.atoms[-1][1].position) < 1e-6
    # atoms outside the sphere should move
    assert abs(np.sum(md.atoms[0][4].position - md.atoms[-1][4].position)) > 1e-6


def test_locality_test(proj_path, cu_box):
    """Test that the 'AnalyseSingleForceSensitivity' runs.

    Notes: the forces on the frozen atom are zero with EMT.
        This should be replaced by some other force field eventually.
        For now this is just a test that the code runs.
    """
    ase.io.write("cu_box.xyz", cu_box)
    thermostat = ips.calculators.LangevinThermostat(
        time_step=1,
        temperature=100,
        friction=1,
    )

    constraints = [
        ips.calculators.FixedSphereConstraint(
            atom_id=0,
            radius=1.0,
        ),
        ips.calculators.FixedSphereConstraint(
            atom_id=0,
            radius=3.0,
        ),
    ]

    with ips.Project(automatic_node_names=True) as project:
        data = ips.AddData(file="cu_box.xyz")
        model = ips.calculators.EMTSinglePoint(data=data.atoms)
        md1 = ips.calculators.ASEMD(
            data=data.atoms,
            model=model,
            thermostat=thermostat,
            init_temperature=1.0,
            steps=30,
            sampling_rate=1,
            dump_rate=33,
            constraint_list=[constraints[0]],
        )
        md2 = ips.calculators.ASEMD(
            data=data.atoms,
            model=model,
            thermostat=thermostat,
            init_temperature=1.0,
            steps=30,
            sampling_rate=1,
            dump_rate=33,
            constraint_list=[constraints[1]],
        )

        ips.analysis.AnalyseSingleForceSensitivity(
            data=[md1, md2],
            sim_list=[md1, md2],
        )

    project.run()
