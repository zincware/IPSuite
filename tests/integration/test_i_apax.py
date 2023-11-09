import pathlib
import shutil

import numpy as np

import ipsuite as ips
from ipsuite.configuration_selection.uniform_energetic import UniformEnergeticSelection
from ipsuite.models import Apax

TEST_PATH = pathlib.Path(__file__).parent.resolve()


def test_apax_model_training(proj_path, traj_file):
    shutil.copy(TEST_PATH / "apax_minimal.yaml", proj_path / "apax_minimal.yaml")

    with ips.Project() as project:
        raw_data = ips.AddData(file=traj_file, name="raw_data")
        train_selection = UniformEnergeticSelection(
            data=raw_data.atoms, n_configurations=10, name="data"
        )

        val_selection = UniformEnergeticSelection(
            data=train_selection.excluded_atoms, n_configurations=8, name="val_data"
        )

        model = Apax(
            config="apax_minimal.yaml",
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
    assert np.any(atoms[0].get_forces() != 0.0)

    analysis.load()

    assert analysis.energy and analysis.forces
    assert not analysis.stress

    for val in analysis.energy.values():
        assert isinstance(val, float)
    for val in analysis.forces.values():
        assert isinstance(val, float)


def test_apax_ensemble(proj_path, traj_file):
    shutil.copy(TEST_PATH / "apax_minimal.yaml", proj_path / "apax_minimal.yaml")
    shutil.copy(TEST_PATH / "apax_minimal2.yaml", proj_path / "apax_minimal2.yaml")

    thermostat = ips.calculators.LangevinThermostat(
        time_step=1.0, temperature=100.0, friction=0.01
    )

    with ips.Project(automatic_node_names=True) as project:
        raw_data = ips.AddData(file=traj_file, name="raw_data")
        train_selection = UniformEnergeticSelection(
            data=raw_data.atoms, n_configurations=10, name="data"
        )

        val_selection = UniformEnergeticSelection(
            data=train_selection.excluded_atoms, n_configurations=8, name="val_data"
        )

        model1 = Apax(
            config="apax_minimal.yaml",
            data=train_selection.atoms,
            validation_data=val_selection.atoms,
        )

        model2 = Apax(
            config="apax_minimal2.yaml",
            data=train_selection.atoms,
            validation_data=val_selection.atoms,
        )

        ensemble_model = ips.models.ApaxEnsemble([model1, model2])

        md = ips.calculators.ASEMD(
            data=raw_data.atoms,
            model=ensemble_model,
            thermostat=thermostat,
            steps=20,
            sampling_rate=1,
        )

        uncertainty_selection = ips.configuration_selection.ThresholdSelection(
            data=md, n_configurations=1, threshold=0.0001
        )

        selection_batch_size = 3
        kernel_selection = ips.models.apax.BatchKernelSelection(
            data=val_selection.atoms,
            train_data=train_selection.atoms,
            models=[model1, model2],
            selection_batch_size = selection_batch_size,
            processing_batch_size=4,
        )

        prediction = ips.analysis.Prediction(data=raw_data, model=ensemble_model)
        prediction_metrics = ips.analysis.PredictionMetrics(data=prediction)

    project.run()

    uncertainty_selection.load()
    kernel_selection.load()
    md.load()

    uncertainties = [x.calc.results["energy_uncertainty"] for x in md.atoms]
    assert [md.atoms[np.argmax(uncertainties)]] == uncertainty_selection.atoms

    assert len(kernel_selection.atoms) == selection_batch_size
