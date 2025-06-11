import os

import ipsuite as ips

os.environ["OPENBLAS_NUM_THREADS"] = "1"


def test_calibration(data_repo):
    water = ips.AddDataH5MD.from_rev(
        name="water", remote="https://github.com/IPSProjects/ips-examples"
    )

    model1 = ips.LJSinglePoint(sigma=1.1)
    model2 = ips.LJSinglePoint(sigma=0.9)

    with ips.Project() as project:
        test_data = ips.RandomSelection(data=water.frames, n_configurations=5)

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
