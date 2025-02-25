import pathlib
import shutil

import ipsuite as ips


def test_lj_single_point(proj_path, traj_file):
    traj_file = pathlib.Path(traj_file)
    shutil.copy(traj_file, ".")

    lj = ips.LJSinglePoint()

    with ips.Project() as project:
        data = ips.AddData(file=traj_file.name)
        pred = ips.Prediction(data=data.frames, model=lj)

    project.repro()

    results = pred.frames[0].calc.results

    assert "energy" in results.keys()
    assert "forces" in results.keys()
