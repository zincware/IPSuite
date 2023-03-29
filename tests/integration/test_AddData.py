import pathlib
import shutil
import subprocess

import ase
import numpy.testing as npt
import pytest

import ipsuite
import znh5md


@pytest.mark.parametrize("eager", [True, False])
def test_AddData(proj_path, traj_file, atoms_list, eager):
    # file would be external otherwise
    traj_file = pathlib.Path(traj_file)
    shutil.copy(traj_file, ".")

    subprocess.check_call(["dvc", "add", traj_file.name])
    with ipsuite.Project() as project:
        data = ipsuite.AddData(file=traj_file.name)

    project.run(eager=eager)
    if not eager:
        data.load()

    assert isinstance(data.atoms, list)
    assert isinstance(data.atoms[0], ase.Atoms)

    for loaded, given in zip(data.atoms[:], atoms_list):
        # Check that the atoms match
        assert loaded.get_potential_energy() == given.get_potential_energy()
        npt.assert_almost_equal(loaded.get_forces(), given.get_forces())
        # use almost equal,
        # because saving to disk will ever so slightly change the forces.


def test_AddDataH5MD(proj_path, atoms_list):
    # file would be external otherwise
    db = znh5md.io.DataWriter("data.h5")
    db.initialize_database_groups()
    db.add(znh5md.io.AtomsReader(atoms_list))
    # shutil.copy(traj_file, ".")

    # subprocess.check_call(["dvc", "add", traj_file.name])
    with ipsuite.Project() as project:
        data = ipsuite.data_loading.AddDataH5MD(file="data.h5")

    project.run()
    data.load()

    assert isinstance(data.atoms, list)
    assert isinstance(data.atoms[0], ase.Atoms)

    for loaded, given in zip(data.atoms[:], atoms_list):
        # Check that the atoms match
        assert loaded.get_potential_energy() == given.get_potential_energy()
        npt.assert_almost_equal(loaded.get_forces(), given.get_forces())
        # use almost equal,
        # because saving to disk will ever so slightly change the forces.
