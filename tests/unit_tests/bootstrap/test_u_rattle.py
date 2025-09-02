import pathlib
import shutil

import pytest

import ipsuite as ips


@pytest.mark.parametrize("include_original", [True, False])
def test_rattle_atoms(proj_path, traj_file, include_original):
    traj_file = pathlib.Path(traj_file)
    shutil.copy(traj_file, ".")

    n_configurations = 10

    with ips.Project() as project:
        data = ips.AddData(file=traj_file.name)

        rattle = ips.RattleAtoms(
            data=data.frames,
            maximum=0.1,
            n_configurations=n_configurations,
            include_original=include_original,
            seed=0,
        )
    project.repro()

    data.load()
    rattle.load()
    rattled_atoms = rattle.frames

    desired_num_configs = n_configurations
    if include_original:
        desired_num_configs += 1

    assert len(rattled_atoms) == desired_num_configs


@pytest.mark.parametrize("include_original", [True, False])
def test_translate_molecules(proj_path, traj_file, include_original):
    traj_file = pathlib.Path(traj_file)
    shutil.copy(traj_file, ".")

    n_configurations = 10

    with ips.Project() as project:
        data = ips.AddData(file=traj_file.name)

        rattle = ips.TranslateMolecules(
            data=data.frames,
            maximum=0.1,
            n_configurations=n_configurations,
            include_original=include_original,
            seed=0,
        )
    project.repro()

    data.load()
    rattle.load()
    rattled_atoms = rattle.frames

    desired_num_configs = n_configurations
    if include_original:
        desired_num_configs += 1

    assert len(rattled_atoms) == desired_num_configs


@pytest.mark.parametrize("include_original", [True, False])
def test_rotate_molecules(proj_path, traj_file, include_original):
    traj_file = pathlib.Path(traj_file)
    shutil.copy(traj_file, ".")

    n_configurations = 10

    with ips.Project() as project:
        data = ips.AddData(file=traj_file.name)

        rattle = ips.RotateMolecules(
            data=data.frames,
            maximum=0.1,
            n_configurations=n_configurations,
            include_original=include_original,
            seed=0,
        )
    project.repro()
    rattled_atoms = rattle.frames

    desired_num_configs = n_configurations
    if include_original:
        desired_num_configs += 1

    assert len(rattled_atoms) == desired_num_configs
    with pytest.raises(RuntimeError):
        assert rattle.frames[1].get_potential_energy() != 0.0
