import ase.io
import numpy as np
import numpy.testing as npt
from ase import units
import pytest

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
            data=data.frames,
            model=model,
            checks=[check],
            modifiers=[rescale_box, temperature_ramp],
            thermostat=thermostat,
            steps=30,
            sampling_rate=1,
            dump_rate=33,
        )
        mapped_md = ips.ASEMD(
            data=data.frames,
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
            data=data.frames,
            model=model,
            checks=[check],
            modifiers=[rescale_box, temperature_ramp],
            thermostat=barostat,
            steps=30,
            sampling_rate=1,
            dump_rate=33,
        )

    project.repro()

    assert len(md.frames) == 30
    assert md.frames[0].cell[0, 0] == 7.22
    assert md.frames[1].cell[0, 0] > 7.22
    assert md.frames[1].cell[0, 0] < 10
    assert md.frames[-1].cell[0, 0] == 10

    assert len(mapped_md.frames) == 30 * 3
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

    model = ips.EMTSinglePoint()
    with ips.Project() as project:
        data = ips.AddData(file="cu_box.xyz")
        md = ips.ASEMD(
            data=data.frames,
            model=model,
            checks=[check],
            modifiers=[rescale_box],
            thermostat=thermostat,
            steps=30,
            sampling_rate=1,
            dump_rate=33,
        )

    project.repro()

    npt.assert_almost_equal(get_density_from_atoms(md.frames[0]), 8971.719659196913)
    npt.assert_almost_equal(get_density_from_atoms(md.frames[-1]), 1000)


def test_ase_md_box_ramp(proj_path, cu_box):
    cu_box[0].set_cell([10,10,10, 90, 90, 90], True)

    ase.io.write("cu_box.xyz", cu_box[0])
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
    model = ips.EMTSinglePoint()
    with ips.Project() as project:
        data = ips.AddData(file="cu_box.xyz")
        md = ips.ASEMD(
            data=data.frames,
            model=model,
            modifiers=[rescale_box],
            thermostat=thermostat,
            steps=20,
            sampling_rate=1,
            dump_rate=33,
        )

    project.repro()

    assert len(md.frames) == 20
    assert md.frames[0].cell[0][0] == 10.0
    assert md.frames[5].cell[0][0] == pytest.approx(12.0, abs=0.1)
    assert md.frames[10].cell[0][0] == pytest.approx(10.0, abs=0.5)
    assert md.frames[15].cell[0][0] == pytest.approx(8.0, abs=0.1)
    assert md.frames[-1].cell[0][0] == 10.0


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
    model = ips.EMTSinglePoint()
    with ips.Project() as project:
        data = ips.AddData(file="cu_box.xyz")
        md = ips.ASEMD(
            data=data.frames,
            model=model,
            modifiers=[temperature_ramp],
            thermostat=thermostat,
            steps=30,
            sampling_rate=1,
            dump_rate=33,
        )

    project.repro()

    assert len(md.frames) == 30
    assert md.frames[0].cell[0, 0] == 7.22
    cell = md.frames[-1].cell
    assert np.all(np.diag(cell.array) - cell[0, 0] < 1e-6)
    assert abs(md.frames[1].cell[0, 0] - 7.22) > 1e-6


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

    model = ips.EMTSinglePoint()
    with ips.Project() as project:
        data = ips.AddData(file="cu_box.xyz")
        md = ips.ASEMD(
            data=data.frames,
            model=model,
            thermostat=thermostat,
            steps=30,
            sampling_rate=1,
            dump_rate=33,
            constraints=[constraint],
        )

    project.repro()

    assert np.sum(md.frames[0][0].position - md.frames[-1][0].position) < 1e-6
    # neighbor atoms should not move
    assert np.sum(md.frames[0][1].position - md.frames[-1][1].position) < 1e-6
    # atoms outside the sphere should move
    assert abs(np.sum(md.frames[0][4].position - md.frames[-1][4].position)) > 1e-6


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

    model = ips.EMTSinglePoint()
    with ips.Project() as project:
        data = ips.AddData(file="cu_box.xyz")
        md1 = ips.ASEMD(
            data=data.frames,
            model=model,
            thermostat=thermostat,
            steps=30,
            sampling_rate=1,
            dump_rate=33,
            constraints=[constraints[0]],
        )
        md2 = ips.ASEMD(
            data=data.frames,
            model=model,
            thermostat=thermostat,
            steps=30,
            sampling_rate=1,
            dump_rate=33,
            constraints=[constraints[1]],
        )

        ips.AnalyseSingleForceSensitivity(
            data=[md1.frames, md2.frames],
            sim_list=[md1, md2],
        )

    project.run()


def test_ase_md_FixedBondLengthConstraint(proj_path):
    thermostat = ips.LangevinThermostat(
        time_step=1,
        temperature=100,
        friction=1,
    )
    model = ips.calculators.EMTSinglePoint()
    constraint = ips.FixedBondLengthConstraint(atom_id_1 = 0, atom_id_2 = 1)

    with ips.Project() as project:
        hydroxide = ips.Smiles2Atoms(smiles="[OH-]")
        md = ips.ASEMD(
            data=hydroxide.frames,
            model=model,
            steps=5,
            thermostat=thermostat,
            sampling_rate=1,
            dump_rate=33,
            constraints=[constraint],
        )

    project.repro()

    d1 = np.linalg.norm(md.frames[0][0].position - md.frames[0][1].position)
    d2 = np.linalg.norm(md.frames[-1][0].position - md.frames[-1][1].position)
    assert np.abs(d2 - d1) < 1e-6


def test_ase_md_safe_reset_modifier(proj_path, cu_box):

    cu_box[0].set_cell([10,10,10, 90, 90, 90], True)

    ase.io.write("cu_box.xyz", cu_box[0])
    check = ips.DebugCheck(n_iterations=10)
    thermostat = ips.LangevinThermostat(
        time_step=1,
        temperature=1,
        friction=1,
    )
    rescale_box = ips.RescaleBoxModifier(cell=100)
    model = ips.LJSinglePoint()

    with ips.Project() as project:
        data = ips.AddData(file="cu_box.xyz")
        md = ips.ASEMDSafeSampling(
            data=data.frames,
            model=model,
            checks=[check],
            modifiers=[rescale_box],
            thermostat=thermostat,
            steps=20,
            sampling_rate=1,
            dump_rate=33,
        )

    project.run()

    setup_box = cu_box[0].cell.diagonal().sum()
    assert setup_box == 30

    # first MD
    npt.assert_almost_equal(md.frames[0].cell.diagonal().sum(), 30)
    # total steps is 20  - 1, as we start counting from 0
    npt.assert_almost_equal(md.frames[9].cell.diagonal().sum(), (10 * 10/19 + 9/19 * 100) * 3)
    
    # second MD runs to the end
    npt.assert_almost_equal(md.frames[10].cell.diagonal().sum(), 30)
    npt.assert_almost_equal(md.frames[-1].cell.diagonal().sum(), 300)

    assert len(md.frames) == 20


def test_ase_md_debug_check(proj_path, cu_box):

    cu_box[0].set_cell([10,10,10, 90, 90, 90], True)

    ase.io.write("cu_box.xyz", cu_box[0])
    check = ips.DebugCheck(n_iterations=10)
    thermostat = ips.LangevinThermostat(
        time_step=1,
        temperature=1,
        friction=1,
    )
    model = ips.LJSinglePoint()

    with ips.Project() as project:
        data = ips.AddData(file="cu_box.xyz")
        md = ips.ASEMD(
            data=data.frames,
            model=model,
            checks=[check],
            thermostat=thermostat,
            steps=20,
            sampling_rate=1,
            dump_rate=33,
        )

    project.run()

    # first frame already has integrated positions
    assert not np.array_equal(md.frames[0].positions, cu_box[0].positions)
    assert len(md.frames) == 10

