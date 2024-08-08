import ipsuite as ips


def test_calibration(data_repo):

    water = ips.data_loading.AddDataH5MD.from_rev(name="water")

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

        ensemble_model = ips.models.EnsembleModel(models=[model1, model2])

        pred = ips.analysis.Prediction(
            data=test_data.atoms,
            model= ensemble_model,
        )
        calibration = ips.analysis.CalibrationMetrics(
            x=test_data.atoms,
            y=pred.atoms,
        )