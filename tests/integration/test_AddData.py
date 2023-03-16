import pathlib
import shutil
import subprocess

import ase
import numpy.testing as npt
import pytest
import znslice

import ipsuite


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
