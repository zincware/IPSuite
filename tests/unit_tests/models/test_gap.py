import os
import pathlib
import random
from pathlib import Path

import ase
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read

from ipsuite.models.base import Prediction
from ipsuite.models.gap import GAP


@pytest.fixture(scope="session", autouse=True)
def traj_file_1(tmp_path_factory) -> str:
    temporary_path = tmp_path_factory.getbasetemp()

    atoms = [
        ase.Atoms(
            "CO", positions=[(0, 0, 0), (0, 0, random.random())], cell=(1, 1, 1), pbc=True
        )
        for _ in range(30)
    ]
    for i, atom in enumerate(atoms):
        atom.calc = SinglePointCalculator(
            atoms=atom, energy=random.random(), forces=np.random.rand(2, 3)
        )

    file = temporary_path / "trajectory.extxyz"
    ase.io.write(file, atoms)

    return file.as_posix()


def test_gap_input():
    gap = GAP(use_energy=True, use_forces=True, use_stresses=True, data=None)
    reference_string = """gap_fite0_method=averageat_file=nodes/MLModel/train_atoms.extxyz
    gap={distance_Nborder=2cutoff=6.0covariance_type=ard_sedelta=0.1
    sparse_method=CUR_POINTSadd_species=Truen_sparse=50theta_uniform=1.0:soapl_max=7
    n_max=7cutoff=6.0atom_sigma=0.5zeta=4.0cutoff_transition_width=0.5delta=1.0
    covariance_type=dot_productn_sparse=50sparse_method=CUR_POINTSadd_species=True}
    gp_file=nodes/MLModel/model/model.xmldefault_sigma={0.00010.010.010.01}
    sparse_jitter=1e-10energy_parameter_name=energy
    force_parameter_name=forcesstress_parameter_name=stress
    virial_parameter_name=DUMMY>>nodes/MLModel/model/GAP_dump.txt"""
    assert gap.gap_input[0].replace(" ", "") == reference_string.replace(" ", "").replace(
        "\n", ""
    )


def test_write_gap_training_data(traj_file_1, tmp_path):
    os.chdir(tmp_path)
    atoms = list(read(traj_file_1, ":10"))
    gap = GAP(data=None)
    gap.write_data_to_file(file=gap.train_data_file, atoms_list=atoms)
    assert gap.train_data_file == pathlib.Path("nodes/MLModel/train_atoms.extxyz")
    assert Path("nodes/MLModel/train_atoms.extxyz").is_file()

    test = list(read(gap.train_data_file, ":"))
    assert len(test) == 10
    assert isinstance(test[0], Atoms)


def test_fit(traj_file_1, tmp_path):
    os.chdir(tmp_path)
    atoms = list(read(traj_file_1, ":10"))
    gap = GAP(data=None)
    gap.data = atoms
    gap.run()
    assert Path(gap.model_directory, "model.xml").is_file()


def test_predict(traj_file_1, tmp_path):
    os.chdir(tmp_path)
    train_atoms = list(read(traj_file_1, ":10"))
    test_atoms = list(read(traj_file_1, "10:20"))
    gap = GAP(data=None)
    gap.use_energy = True
    gap.use_forces = True
    gap.use_stresses = True
    gap.data = train_atoms
    gap.run()
    predicted_values = gap.predict(atoms=test_atoms)
    assert isinstance(predicted_values, Prediction)
    assert np.shape(predicted_values.energy) == (len(test_atoms),)
    assert np.shape(predicted_values.forces) == (len(test_atoms), len(test_atoms[0]), 3)
    assert np.shape(predicted_values.stresses) == (len(test_atoms), 6)
