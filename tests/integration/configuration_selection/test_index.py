"""Use index selection to test the selection base."""

import pytest
import zntrack

import ipsuite as ips


def test_index_chained(proj_path, traj_file):
    with ips.Project() as project:
        data = ips.AddData(file=traj_file)
        pre_selection = ips.IndexSelection(
            data=data.atoms,
            start=0,
            stop=5,
            step=None,
        )  # we use this to "change" the data
        selection = ips.IndexSelection(
            data=pre_selection.atoms, indices=[0, 1, 2], name="selection"
        )

        histogram = ips.EnergyHistogram(data=selection.atoms)

    project.repro()

    assert histogram.labels_df.to_dict()["bin_edges"][0] == pytest.approx(
        0.0952380952380952
    )

    with ips.Project() as project:
        data = ips.AddData(file=traj_file)
        pre_selection = ips.IndexSelection(
            data=data.atoms,
            start=5,
            stop=10,
            step=None,
        )  # we use this to "change" the data
        selection = ips.IndexSelection(
            data=pre_selection.atoms, indices=[0, 1, 2], name="selection"
        )

        histogram = ips.EnergyHistogram(data=selection.atoms)

    project.repro()
    histogram = zntrack.from_rev(name=histogram.name)
    assert histogram.labels_df.to_dict()["bin_edges"][0] == pytest.approx(
        0.3333333333333333
    )


def test_filter_outlier(proj_path, traj_file):
    with ips.Project() as project:
        data = ips.AddData(file=traj_file)
        filtered_data = ips.FilterOutlier(
            data=data.atoms, key="energy", threshold=1, direction="both"
        )

    project.repro()

    filtered_data = zntrack.from_rev(name=filtered_data.name)

    assert len(filtered_data.atoms) == 13
