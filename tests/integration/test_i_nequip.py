import pathlib
import shutil

import numpy as np

import ipsuite

# tests/integration_test/models/test_i_nequip.py == __file__
TEST_PATH = pathlib.Path(__file__).parent.resolve()


def test_model_training(proj_path, traj_file):
    shutil.copy(TEST_PATH / "allegro_minimal.yaml", proj_path / "allegro_minimal.yaml")
    with ipsuite.Project() as project:
        data_1 = ipsuite.AddData(file=traj_file, name="data_1")

        train_selection = ipsuite.configuration_selection.UniformEnergeticSelection(
            data=data_1, n_configurations=10, name="train_data"
        )

        validation_selection = ipsuite.configuration_selection.UniformEnergeticSelection(
            data=train_selection @ "excluded_atoms", n_configurations=8, name="val_data"
        )

        model = ipsuite.models.Nequip(
            parameter="allegro_minimal.yaml",
            data=train_selection,
            validation_data=validation_selection,
            device="cpu",
        )

    project.run()

    data_1.load()

    model.load()

    atoms = data_1.atoms[0]
    atoms.calc = model.calc

    assert isinstance(atoms.get_potential_energy(), float)
    assert atoms.get_potential_energy() != 0.0

    assert isinstance(atoms.get_forces(), np.ndarray)
    assert atoms.get_forces()[0, 0] != 0.0

    assert model.lammps_pair_style == "allegro"
    assert model.lammps_pair_coeff[0] == "* * nodes/MLModel/deployed_model.pth C O"
