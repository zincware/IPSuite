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

        rattle = ips.bootstrap.RattleAtoms(
            data=data.atoms,
            maximum=0.1,
            n_configurations=n_configurations,
            include_original=include_original,
            seed=0,
        )
    project.run()

    data.load()
    rattle.load()
    rattled_atoms = rattle.atoms

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

        rattle = ips.bootstrap.TranslateMolecules(
            data=data.atoms,
            maximum=0.1,
            n_configurations=n_configurations,
            include_original=include_original,
            seed=0,
        )
    project.run()

    data.load()
    rattle.load()
    rattled_atoms = rattle.atoms

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

        rattle = ips.bootstrap.RotateMolecules(
            data=data.atoms,
            maximum=0.1,
            n_configurations=n_configurations,
            include_original=include_original,
            seed=0,
        )
    project.run()

    data.load()
    rattle.load()
    rattled_atoms = rattle.atoms

    desired_num_configs = n_configurations
    if include_original:
        desired_num_configs += 1

    assert len(rattled_atoms) == desired_num_configs
    with pytest.raises(RuntimeError):
        assert rattle.atoms[0].get_potential_energy() != 0.0


def test_rotate_molecules_with_calc(proj_path, traj_file):
    traj_file = pathlib.Path(traj_file)
    shutil.copy(traj_file, ".")

    n_configurations = 10

    with ips.Project() as project:
        data = ips.AddData(file=traj_file.name)

        model = ips.calculators.EMTSinglePoint(data=None)

        rattle = ips.bootstrap.RotateMolecules(
            data=data.atoms,
            maximum=0.1,
            n_configurations=n_configurations,
            include_original=False,
            seed=0,
            model=model,
        )
    project.repro()

    # assert all entries in the atoms[x] list have a different potential energy

    energies = [atoms.get_potential_energy() for atoms in rattle.atoms]
    assert len(set(energies)) == len(energies)  # all different
    assert energies[0] != 0.0
