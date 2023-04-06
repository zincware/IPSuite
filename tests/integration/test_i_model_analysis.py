import pathlib
import random

import ase.io
import numpy as np
import pytest
from ase.calculators.singlepoint import SinglePointCalculator

import ipsuite

TEST_PATH = pathlib.Path(__file__).parent.resolve()


@pytest.fixture()
def trained_model(proj_path, traj_file) -> tuple:
    with ipsuite.Project() as project:
        data_1 = ipsuite.AddData(file=traj_file, name="data_1")

        train_selection = ipsuite.configuration_selection.UniformEnergeticSelection(
            data=data_1, n_configurations=10, name="train_data"
        )

        validation_selection = ipsuite.configuration_selection.UniformEnergeticSelection(
            data=train_selection @ "excluded_atoms", n_configurations=8, name="val_data"
        )

        model = ipsuite.models.GAP(soap={"cutoff": 0.7}, data=train_selection)

    project.run()

    return project, model, validation_selection


@pytest.mark.parametrize("eager", [True, False])
def test_PredictWithModel(trained_model, eager):
    project, model, validation_selection = trained_model

    with project:
        analysis = ipsuite.analysis.Prediction(model=model, data=validation_selection)
    project.run(eager=eager)
    if not eager:
        analysis.load()

    assert np.any(
        np.not_equal(analysis.data[0].get_forces(), analysis.atoms[0].get_forces())
    )
    assert isinstance(analysis.data[0], ase.Atoms)
    assert isinstance(analysis.atoms[0], ase.Atoms)

    for atoms in analysis.atoms:
        assert atoms.get_potential_energy() != 0.0
        assert not np.isnan(atoms.get_potential_energy())

    for atoms in analysis.data:
        assert atoms.get_potential_energy() != 0.0
        assert not np.isnan(atoms.get_potential_energy())

    assert analysis.data[0].get_forces().shape == (2, 3)
    assert analysis.atoms[0].get_forces().shape == (2, 3)


@pytest.mark.parametrize("eager", [True, False])
def test_AnalysePrediction(trained_model, eager):
    project, model, validation_selection = trained_model

    with project:
        prediction = ipsuite.analysis.Prediction(model=model, data=validation_selection)
        analysis = ipsuite.analysis.Metrics(data=prediction)
    project.run(eager=eager)
    if not eager:
        analysis.load()

    assert not analysis.energy_df.empty
    assert analysis.energy["rmse"] > 0.0


@pytest.mark.parametrize("eager", [True, False])
def test_AnalyseForceAngles(trained_model, eager):
    project, model, validation_selection = trained_model
    with project:
        prediction = ipsuite.analysis.Prediction(model=model, data=validation_selection)
        analysis = ipsuite.analysis.ForceAngles(data=prediction)

    project.run(eager=eager)
    if not eager:
        analysis.load()

    assert analysis.plot.exists()
    assert analysis.log_plot.exists()
    assert analysis.angles["rmse"] > 0.0


def test_RattleAnalysis(trained_model):
    project, model, validation_selection = trained_model

    with project:
        analysis = ipsuite.analysis.RattleAtoms(model=model, data=validation_selection)
    project.run()

    analysis.load()
    assert analysis.energies is not None


def test_BoxScaleAnalysis(trained_model):
    project, model, validation_selection = trained_model

    with project:
        analysis = ipsuite.analysis.BoxScale(
            model=model, data=validation_selection, num=10, stop=1.1
        )
    project.run()
    analysis.load()

    assert analysis.energies is not None


@pytest.mark.parametrize("eager", [True, False])
def test_MDStabilityAnalysis(trained_model, eager):
    project, model, validation_selection = trained_model

    checks = [
        ipsuite.analysis.NaNCheck(),
        ipsuite.analysis.ConnectivityCheck(),
        ipsuite.analysis.EnergySpikeCheck(min_factor=0.5, max_factor=2.0),
    ]
    with project:
        analysis = ipsuite.analysis.MDStability(
            model=model,
            data=validation_selection.atoms,
            max_steps=500,
            time_step=0.05,
            checks=checks,
            bins=10,
            save_last_n=1,
        )
    project.run(eager=eager)

    if not eager:
        analysis.load()

    validation_selection.load()
    assert len(analysis.atoms) == len(validation_selection.atoms)
