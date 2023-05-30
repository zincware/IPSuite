import pathlib
import shutil

import numpy as np

import ipsuite as ips
from ipsuite.configuration_selection.uniform_energetic import UniformEnergeticSelection
from ipsuite.models import Apax

TEST_PATH = pathlib.Path(__file__).parent.resolve()


def test_model_training(proj_path, traj_file):
    shutil.copy(TEST_PATH / "apax_minimal.yaml", proj_path / "apax_minimal.yaml")

    rev_forces = np.array([[-0.0, -0.0, -0.00127511], [-0.0, -0.0, 0.00127511]])
    with ips.Project() as project:
        raw_data = ips.AddData(file=traj_file, name="raw_data")
        train_selection = UniformEnergeticSelection(
            data=raw_data.atoms, n_configurations=10, name="data"
        )

        val_selection = UniformEnergeticSelection(
            data=train_selection.excluded_atoms, n_configurations=8, name="val_data"
        )

        model = Apax(
            parameter_file="apax_minimal.yaml",
            data=train_selection.atoms,
            validation_data=val_selection.atoms,
        )

        prediction = ips.analysis.Prediction(model=model, data=val_selection.atoms)
        analysis = ips.analysis.PredictionMetrics(data=prediction)

    project.run()

    raw_data.load()
    atoms = raw_data.atoms

    model.load()
    atoms[0].calc = model.get_calculator()

    assert isinstance(atoms[0].get_potential_energy(), float)
    assert atoms[0].get_potential_energy() != 0.0

    assert isinstance(atoms[0].get_forces(), np.ndarray)
    assert atoms[0].get_forces().shape == (2, 3)
    assert np.all(atoms[0].get_forces()[:, 0:2] == np.full([2, 2], 0.0))
    assert np.all(atoms[0].get_forces()[:, 2] != np.full([2, 1], 0.0))

    analysis.load()

    assert analysis.energy and analysis.forces
    assert not analysis.stress

    for val in analysis.energy.values():
        assert isinstance(val, float)
    for val in analysis.forces.values():
        assert isinstance(val, float)
