"""Collection of common fixtures.

References
----------
https://docs.pytest.org/en/6.2.x/fixture.html#scope-sharing-fixtures-across-classes-modules-packages-or-session
"""

import logging
import os
import pathlib
import random
import shutil
import typing

import ase
import ase.calculators.singlepoint
import ase.io
import dvc.cli
import git
import numpy as np
import pytest
import zntrack
from ase import Atoms
from ase.lattice.cubic import FaceCenteredCubic

import ipsuite as ips

zntrack.config.log_level = logging.DEBUG


@pytest.fixture
def atoms_list() -> typing.List[ase.Atoms]:
    """Generate ase.Atoms objects.

    Construct Atoms objects with random positions and increasing energy
    and random force values.
    """
    random.seed(1234)
    atoms = [
        ase.Atoms(
            "CO",
            positions=[(0, 0, 0), (0, 0, random.random())],
            cell=(1, 1, 1),
            pbc=True,
        )
        for _ in range(21)
    ]

    for idx, atom in enumerate(atoms):
        atom.calc = ase.calculators.singlepoint.SinglePointCalculator(
            atoms=atom,
            energy=idx / 21,
            forces=np.random.randn(2, 3),
            stress=np.random.randn(6),
            energy_uncertainty=idx + 2,
            forces_uncertainty=np.full((2, 3), 2.0) + idx,
        )

    return atoms


@pytest.fixture
def cu_box() -> typing.List[ase.Atoms]:
    return [
        FaceCenteredCubic(
            directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            symbol="Cu",
            size=(2, 2, 2),
            pbc=True,
        )
    ]


@pytest.fixture()
def traj_file(tmp_path_factory, atoms_list) -> str:
    """Save an extxyz trajectory file based on atoms_list."""
    temporary_path = tmp_path_factory.getbasetemp()
    file = temporary_path / "trajectory.extxyz"
    ase.io.write(file, atoms_list)

    return file.as_posix()


@pytest.fixture
def proj_path(tmp_path, request) -> pathlib.Path:
    """Temporary directory for testing DVC calls.

    Parameters
    ----------
    tmp_path:
        temporary directory
    request:
        https://docs.pytest.org/en/6.2.x/reference.html#std-fixture-request
    Returns
    -------
    path to temporary directory.
    """
    shutil.copy(request.module.__file__, tmp_path)
    os.chdir(tmp_path)

    git.Repo.init()
    dvc.cli.main(["init"])

    return tmp_path


@pytest.fixture
def proj_w_data(proj_path, traj_file, request) -> typing.Tuple[ips.Project, ips.AddData]:
    data = []
    with ips.Project() as proj:
        for idx in range(request.param):
            data.append(ips.AddData(file=traj_file, name=f"data_{idx}"))
    proj.run()
    return proj, data


@pytest.fixture
def data_repo(tmp_path, request) -> pathlib.Path:
    git.Repo.clone_from(
        r"https://dagshub.com/PythonFZ/IPS_test_data.git", tmp_path, branch="znh5md_fix"
    )
    shutil.copy(request.module.__file__, tmp_path)
    os.chdir(tmp_path)
    dvc.cli.main(["pull"])
    # we must pull here, because the HTTP remote is not supported by DVCFileSystem
    # and S3 requires credentials.

    return tmp_path


@pytest.fixture
def atoms_with_composed_forces():
    atoms = Atoms(
        "OH2",
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
    )
    ft = np.array(
        [
            [0.0, 0.0, 1.0 * 15.999],
            [0.0, 0.0, 1.008],
            [0.0, 0.0, 1.008],
        ]
    )
    fr = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.008],
            [0.0, 0.0, -1.008],
        ]
    )
    fv = np.array(
        [
            [1.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    atoms.calc = ase.calculators.singlepoint.SinglePointCalculator(
        atoms, forces=ft + fr + fv
    )

    return atoms, ft, fr, fv
