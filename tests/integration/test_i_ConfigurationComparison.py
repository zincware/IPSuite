import pathlib

import pytest

import ipsuite


@pytest.mark.parametrize(
    "method",
    [
        ipsuite.MMKernel,
        ipsuite.REMatch,
    ],
)
def test_ConfigurationComparison(proj_path, traj_file, method):
    with ipsuite.Project() as project:
        data_1 = ipsuite.AddData(file=traj_file, name="data_1")
        comparison = method(analyte=data_1)
    project.repro()

    comparison.load()
    df_max = comparison.similarities
    assert pathlib.Path(
        proj_path, "nodes/ConfigurationComparison/similarities.csv"
    ).is_file()
    assert len(df_max["similarities"]) == 20
    assert 0 <= (df_max["similarities"].sum() / len(df_max["similarities"])) <= 1.0
