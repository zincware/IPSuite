import ase
import numpy as np

import ipsuite as ips


def test_ensemble_model(data_repo):
    water = ips.data_loading.AddDataH5MD.from_rev(name="water")

    thermostat = ips.calculators.LangevinThermostat(
        time_step=1.0, temperature=100.0, friction=0.01
    )

    with ips.Project(automatic_node_names=True) as project:
        test_data = ips.configuration_selection.RandomSelection(
            data=water.atoms, n_configurations=5
        )

        train_data = ips.configuration_selection.RandomSelection(
            data=water.atoms,
            n_configurations=5,
            exclude_configurations=test_data.selected_configurations,
        )

        model1 = ips.models.GAP(data=train_data.atoms, soap={"n_max": 2})
        model2 = ips.models.GAP(data=train_data.atoms, soap={"n_max": 3})
        model3 = ips.models.GAP(data=train_data.atoms, soap={"n_max": 4})

        ensemble_model = ips.models.EnsembleModel(models=[model1, model2, model3])

        md = ips.calculators.ASEMD(
            data=test_data.atoms,
            model=ensemble_model,
            thermostat=thermostat,
            steps=20,
            sampling_rate=1,
        )

        energy_uncertainty_hist = ips.analysis.EnergyUncertaintyHistogram(data=md.atoms)
        forces_uncertainty_hist = ips.analysis.ForcesUncertaintyHistogram(data=md.atoms)

        uncertainty_selection = ips.configuration_selection.ThresholdSelection(
            data=md, n_configurations=1, threshold=0.0001
        )

        ips.analysis.ModelEnsembleAnalysis(
            data=test_data.atoms, models=[model1, model2, model3]
        )

        prediction = ips.analysis.Prediction(data=test_data.atoms, model=ensemble_model)
        prediction_metrics = ips.analysis.PredictionMetrics(
            true_data=test_data.atoms, pred_data=prediction.atoms
        )

    project.run(environment={"OPENBLAS_NUM_THREADS": "1"})

    uncertainty_selection.load()
    md.load()

    uncertainties = [x.calc.results["energy_uncertainty"] for x in md.atoms]
    assert [md.atoms[np.argmax(uncertainties)]] == uncertainty_selection.atoms


def test_ensemble_model_stress(proj_path, cu_box):
    ase.io.write("cu_box.xyz", cu_box)

    with ips.Project(automatic_node_names=True) as project:
        data = ips.AddData(file="cu_box.xyz")
        model1 = ips.calculators.EMTSinglePoint(data=data.atoms)
        model2 = ips.calculators.EMTSinglePoint(data=data.atoms)
        ensemble_model = ips.models.EnsembleModel(models=[model1, model2])

        prediction = ips.analysis.Prediction(model=ensemble_model, data=model1.atoms)
        analysis = ips.analysis.PredictionMetrics(
            true_data=model1.atoms, pred_data=prediction.atoms
        )

    project.run(eager=False)

    analysis.load()

    assert (len(analysis.content["stress_pred"])) > 0
    assert (len(analysis.content["stress_hydro_pred"])) > 0
    assert (len(analysis.content["stress_deviat_pred"])) > 0
