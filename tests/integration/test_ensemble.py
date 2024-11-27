import os

import ase
import numpy as np
import zntrack

import ipsuite as ips

os.environ["OPENBLAS_NUM_THREADS"] = "1"


def test_ensemble_model(proj_path, traj_file):
    data = ips.AddDataH5MD.from_rev(
        name="water", remote="https://github.com/IPSProjects/ips-examples"
    )
    thermostat = ips.LangevinThermostat(time_step=1.0, temperature=100.0, friction=0.01)

    with ips.Project() as project:
        test_data = ips.RandomSelection(data=data.atoms, n_configurations=5)

        train_data = ips.RandomSelection(
            data=test_data.excluded_atoms,
            n_configurations=5,
        )

        model1 = ips.GAP(data=train_data.atoms, soap={"n_max": 1}, use_stresses=False)
        model2 = ips.GAP(data=train_data.atoms, soap={"n_max": 2}, use_stresses=False)

        ensemble_model = ips.EnsembleModel(models=[model1, model2])

        md = ips.ASEMD(
            data=test_data.atoms,
            model=ensemble_model,
            thermostat=thermostat,
            steps=20,
            sampling_rate=1,
        )

        energy_uncertainty_hist = ips.EnergyUncertaintyHistogram(data=md.atoms)
        forces_uncertainty_hist = ips.ForcesUncertaintyHistogram(data=md.atoms)

        uncertainty_selection = ips.ThresholdSelection(
            data=md.atoms, n_configurations=1, threshold=0.0001
        )

        ips.ModelEnsembleAnalysis(data=test_data.atoms, models=[model1, model2])

        prediction = ips.Prediction(data=test_data.atoms, model=ensemble_model)
        prediction_metrics = ips.PredictionMetrics(x=test_data.atoms, y=prediction.atoms)

    project.repro()

    uncertainties = [x.calc.results["energy_uncertainty"] for x in md.atoms]
    # https://github.com/zincware/ZnTrack/pull/854
    uncertainty_selection = zntrack.from_rev(name=uncertainty_selection.name)
    assert [md.atoms[np.argmax(uncertainties)]] == uncertainty_selection.atoms


def test_ensemble_model_stress(proj_path, traj_file):
    model1 = ips.EMTSinglePoint()
    model2 = ips.EMTSinglePoint()

    with ips.Project() as project:
        data = ips.AddData(file=traj_file)
        ensemble_model = ips.EnsembleModel(models=[model1, model2])

        prediction = ips.Prediction(model=ensemble_model, data=data.atoms)
        analysis = ips.PredictionMetrics(x=data.atoms, y=prediction.atoms)

    project.repro()

    content = analysis.get_content()
    assert (len(content["stress_pred"])) > 0
    assert (len(content["stress_hydro_pred"])) > 0
    assert (len(content["stress_deviat_pred"])) > 0
