import ase.io
import numpy as np
import numpy.testing as npt
import zntrack as zn
from ase import units

import ipsuite as ips
from ipsuite.utils.ase_sim import get_density_from_atoms


def test_ase_run_md(proj_path, cu_box):
    atoms = []
    for _ in range(5):
        atoms.extend(cu_box)

    ase.io.write("cu_box.xyz", atoms)
    check = ips.TemperatureCheck()
    thermostat = ips.LangevinThermostat(
        time_step=1,
        temperature=1,
        friction=1,
    )
    barostat = ips.SVCRBarostat(
        time_step=1,
        temperature=1,
    )
    rescale_box = ips.RescaleBoxModifier(cell=10)
    temperature_ramp = ips.TemperatureRampModifier(temperature=300)
    model = ips.LJSinglePoint()

    with ips.Project() as project:
        data = ips.AddData(file="cu_box.xyz")
        md = ips.ASEMD(
            data=data.atoms,
            model=model,
            checks=[check],
            modifiers=[rescale_box, temperature_ramp],
            thermostat=thermostat,
            steps=30,
            sampling_rate=1,
            dump_rate=33,
        )
        mapped_md = zn.apply(ips.ASEMD, method="map")(
            data=data.atoms,
            data_ids=[0, 1, 2],
            model=model,
            checks=[check],
            modifiers=[rescale_box, temperature_ramp],
            thermostat=thermostat,
            steps=30,
            sampling_rate=1,
            dump_rate=33,
        )
        md2 = ips.ASEMD(
            data=data.atoms,
            model=model,
            checks=[check],
            modifiers=[rescale_box, temperature_ramp],
            thermostat=barostat,
            steps=30,
            sampling_rate=1,
            dump_rate=33,
        )

    project.repro()

    assert len(md.atoms) == 30
    assert md.atoms[0].cell[0, 0] == 7.22
    assert md.atoms[1].cell[0, 0] > 7.22
    assert md.atoms[1].cell[0, 0] < 10
    assert md.atoms[-1].cell[0, 0] == 10

    assert len(mapped_md.atoms) == 30 * 3
    assert len(mapped_md.structures) == 3


def test_ase_md_target_density(proj_path, cu_box):
    ase.io.write("cu_box.xyz", cu_box)
    check = ips.TemperatureCheck()
    thermostat = ips.LangevinThermostat(
        time_step=1,
        temperature=1,
        friction=1,
    )
    rescale_box = ips.RescaleBoxModifier(density=1000)

    with ips.Project() as project:
        data = ips.AddData(file="cu_box.xyz")
        model = ips.EMTSinglePoint(data=data.atoms)
        md = ips.ASEMD(
            data=data.atoms,
            model=model,
            checks=[check],
            modifiers=[rescale_box],
            thermostat=thermostat,
            steps=30,
            sampling_rate=1,
            dump_rate=33,
        )

    project.repro()

    npt.assert_almost_equal(get_density_from_atoms(md.atoms[0]), 8971.719659196913)
    npt.assert_almost_equal(get_density_from_atoms(md.atoms[-1]), 1000)


def test_ase_md_box_ramp(proj_path, cu_box):
    ase.io.write("cu_box.xyz", cu_box)
    thermostat = ips.LangevinThermostat(
        time_step=1,
        temperature=1,
        friction=1,
    )
    rescale_box = ips.BoxOscillatingRampModifier(
        end_cell=10.0,
        cell_amplitude=2.0,
        num_oscillations=1.0,
    )
    with ips.Project() as project:
        data = ips.AddData(file="cu_box.xyz")
        model = ips.EMTSinglePoint(data=data.atoms)
        md = ips.ASEMD(
            data=data.atoms,
            model=model,
            modifiers=[rescale_box],
            thermostat=thermostat,
            steps=20,
            sampling_rate=1,
            dump_rate=33,
        )

    project.repro()

    assert len(md.atoms) == 20
    assert md.atoms[0].cell[0, 0] == 7.22
    assert md.atoms[1].cell[0, 0] > 7.22
    assert md.atoms[1].cell[0, 0] < 10
    assert (md.atoms[-1].cell[0, 0] - 10.0) < 1e-6


def test_ase_npt(proj_path, cu_box):
    ase.io.write("cu_box.xyz", cu_box)
    thermostat = ips.NPTThermostat(
        time_step=1.0,
        temperature=300,
        pressure=1.01325 * units.bar,
        ttime=25 * units.fs,
        pfactor=(75 * units.fs) ** 2,
        tetragonal_strain=True,
        fraction_traceless=0.0,
    )
    temperature_ramp = ips.TemperatureOscillatingRampModifier(
        end_temperature=100.0,
        temperature_amplitude=20.0,
        num_oscillations=2.0,
    )
    with ips.Project() as project:
        data = ips.AddData(file="cu_box.xyz")
        model = ips.EMTSinglePoint(data=data.atoms)
        md = ips.ASEMD(
            data=data.atoms,
            model=model,
            modifiers=[temperature_ramp],
            thermostat=thermostat,
            steps=30,
            sampling_rate=1,
            dump_rate=33,
        )

    project.repro()

    assert len(md.atoms) == 30
    assert md.atoms[0].cell[0, 0] == 7.22
    cell = md.atoms[-1].cell
    assert np.all(np.diag(cell.array) - cell[0, 0] < 1e-6)
    assert abs(md.atoms[1].cell[0, 0] - 7.22) > 1e-6


def test_ase_md_fixed_sphere(proj_path, cu_box):
    ase.io.write("cu_box.xyz", cu_box)
    thermostat = ips.LangevinThermostat(
        time_step=1,
        temperature=1,
        friction=1,
    )

    constraint = ips.FixedSphereConstraint(
        atom_id=0,
        radius=2.6,
    )

    with ips.Project() as project:
        data = ips.AddData(file="cu_box.xyz")
        model = ips.EMTSinglePoint(data=data.atoms)
        md = ips.ASEMD(
            data=data.atoms,
            model=model,
            thermostat=thermostat,
            steps=30,
            sampling_rate=1,
            dump_rate=33,
            constraints=[constraint],
        )

    project.repro()

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
    thermostat = ips.LangevinThermostat(
        time_step=1,
        temperature=100,
        friction=1,
    )

    constraints = [
        ips.FixedSphereConstraint(
            atom_id=0,
            radius=1.0,
        ),
        ips.FixedSphereConstraint(
            atom_id=0,
            radius=3.0,
        ),
    ]

    with ips.Project() as project:
        data = ips.AddData(file="cu_box.xyz")
        model = ips.EMTSinglePoint(data=data.atoms)
        md1 = ips.ASEMD(
            data=data.atoms,
            model=model,
            thermostat=thermostat,
            steps=30,
            sampling_rate=1,
            dump_rate=33,
            constraints=[constraints[0]],
        )
        md2 = ips.ASEMD(
            data=data.atoms,
            model=model,
            thermostat=thermostat,
            steps=30,
            sampling_rate=1,
            dump_rate=33,
            constraints=[constraints[1]],
        )

        ips.AnalyseSingleForceSensitivity(
            data=[md1, md2],
            sim_list=[md1, md2],
        )

    project.repro()
