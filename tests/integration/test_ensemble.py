import ipsuite as ips


def test_ensemble_model(data_repo):
    water = ips.data_loading.AddDataH5MD.from_rev(name="water")

    with ips.Project(automatic_node_names=True) as project:
        test_data = ips.configuration_selection.RandomSelection(
            data=water, n_configurations=5
        )

        train_data = ips.configuration_selection.RandomSelection(
            data=water,
            n_configurations=5,
            exclude_configurations=test_data.selected_configurations,
        )

        model1 = ips.models.GAP(data=train_data, soap={"n_max": 2})
        model2 = ips.models.GAP(data=train_data, soap={"n_max": 3})
        model3 = ips.models.GAP(data=train_data, soap={"n_max": 4})

        ensemble_model = ips.models.EnsembleModel(models=[model1, model2, model3])

        md = ips.calculators.ASEMD(
            data=test_data.atoms,
            model=ensemble_model,
            temperature=100,
            time_step=1.0,
            friction=0.01,
            steps=20,
            sampling_rate=1,
        )

        uncertainty_selection = ips.configuration_selection.ThresholdSelection(
            data=md, n_configurations=5
        )
        # assert that these are the highest uncertainties

        ips.analysis.ModelEnsembleAnalysis(
            data=test_data, models=[model1, model2, model3]
        )

        prediction = ips.analysis.Prediction(data=test_data, model=ensemble_model)
        prediction_metrics = ips.analysis.PredictionMetrics(data=prediction)

    project.run(environment={"OPENBLAS_NUM_THREADS": "1"})

    raise ValueError(data_repo)
