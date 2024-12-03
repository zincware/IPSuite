import os

import ipsuite as ips

os.environ["OPENBLAS_NUM_THREADS"] = "1"


def test_calibration(data_repo):
    water = ips.AddDataH5MD.from_rev(
        name="water", remote="https://github.com/IPSProjects/ips-examples"
    )

    with ips.Project() as project:
        test_data = ips.RandomSelection(data=water.frames, n_configurations=5)

        train_data = ips.RandomSelection(
            data=test_data.excluded_frames,
            n_configurations=5,
        )

        model1 = ips.GAP(data=train_data.frames, soap={"n_max": 1})
        model2 = ips.GAP(data=train_data.frames, soap={"n_max": 2})

        ensemble_model = ips.EnsembleModel(models=[model1, model2])

        pred = ips.Prediction(
            data=test_data.frames,
            model=ensemble_model,
        )
        calibration = ips.CalibrationMetrics(
            x=test_data.frames,
            y=pred.frames,
        )

    project.repro()
