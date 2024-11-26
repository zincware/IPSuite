import os

import ipsuite as ips

os.environ["OPENBLAS_NUM_THREADS"] = "1"


def test_calibration(data_repo):
    water = ips.AddDataH5MD.from_rev(name="water")

    with ips.Project() as project:
        test_data = ips.RandomSelection(
            data=water.atoms, n_configurations=5
        )

        train_data = ips.RandomSelection(
            data=water.atoms,
            n_configurations=5,
            exclude_configurations=test_data.selected_configurations,
        )

        model1 = ips.GAP(data=train_data.atoms, soap={"n_max": 2})
        model2 = ips.GAP(data=train_data.atoms, soap={"n_max": 3})

        ensemble_model = ips.EnsembleModel(models=[model1, model2])

        pred = ips.Prediction(
            data=test_data.atoms,
            model=ensemble_model,
        )
        calibration = ips.CalibrationMetrics(
            x=test_data.atoms,
            y=pred.atoms,
        )

    project.repro()
