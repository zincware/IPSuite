import pathlib
import shutil

import pytest

import ipsuite as ips


@pytest.mark.parametrize("include_original", [True, False])
def test_rattle_atoms(proj_path, traj_file, include_original):
    traj_file = pathlib.Path(traj_file)
    shutil.copy(traj_file, ".")

    n_configs = 10

    with ips.Project() as project:
        data = ips.AddData(file=traj_file.name)

        rattle = ips.bootstrap.RattleAtoms(
            data=data.atoms,
            range_maximum=0.1,
            n_configs=n_configs,
            include_original=include_original,
            seed=0,
        )
    project.run()

    data.load()
    rattle.load()
    rattled_atoms = rattle.atoms

    desired_num_configs = n_configs
    if include_original:
        desired_num_configs += 1

    assert len(rattled_atoms) == desired_num_configs


@pytest.mark.parametrize("include_original", [True, False])
def test_translate_molecules(proj_path, traj_file, include_original):
    traj_file = pathlib.Path(traj_file)
    shutil.copy(traj_file, ".")

    n_configs = 10

    with ips.Project() as project:
        data = ips.AddData(file=traj_file.name)

        rattle = ips.bootstrap.TranslateMolecules(
            data=data.atoms,
            range_maximum=0.1,
            n_configs=n_configs,
            include_original=include_original,
            seed=0,
        )
    project.run()

    data.load()
    rattle.load()
    rattled_atoms = rattle.atoms

    desired_num_configs = n_configs
    if include_original:
        desired_num_configs += 1

    assert len(rattled_atoms) == desired_num_configs


@pytest.mark.parametrize("include_original", [True, False])
def test_rotate_molecules(proj_path, traj_file, include_original):
    traj_file = pathlib.Path(traj_file)
    shutil.copy(traj_file, ".")

    n_configs = 10

    with ips.Project() as project:
        data = ips.AddData(file=traj_file.name)

        rattle = ips.bootstrap.RotateMolecules(
            data=data.atoms,
            range_maximum=0.1,
            n_configs=n_configs,
            include_original=include_original,
            seed=0,
        )
    project.run()

    data.load()
    rattle.load()
    rattled_atoms = rattle.atoms

    desired_num_configs = n_configs
    if include_original:
        desired_num_configs += 1

    assert len(rattled_atoms) == desired_num_configs