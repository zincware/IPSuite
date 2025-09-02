"""pytest setup for `pytest --doctest-modules ipsuite/`."""

import os
import pathlib
import subprocess

import ase.io
import pytest
import rdkit2ase

import ipsuite as ips


@pytest.fixture
def project(tmp_path: pathlib.Path):
    """
    A pytest fixture that creates a temporary directory,
    initializes git and dvc, and yields an ips.Project instance.
    """
    # Store the original directory to return to it later
    original_cwd = pathlib.Path.cwd()
    os.chdir(tmp_path)

    ethanol = rdkit2ase.smiles2conformers("CCO", numConfs=100)
    ase.io.write("ethanol.xyz", ethanol)

    # Setup: Initialize git and DVC
    try:
        subprocess.run(["git", "init"], check=True, capture_output=True)
        # Using --quiet to keep the output clean
        subprocess.run(["dvc", "init", "--quiet"], check=True, capture_output=True)

        # Yield the project instance for the test to use
        yield ips.Project()

    finally:
        # Teardown: Go back to the original directory
        os.chdir(original_cwd)


@pytest.fixture(autouse=True)
def doctest_namespace(project):
    """
    Makes the 'project' fixture and the 'ips' module
    available to all doctests.
    """
    return {"project": project, "ips": ips}
