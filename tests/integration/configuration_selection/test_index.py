"""Use index selection to test the selection base."""

import pytest
import zntrack

import ipsuite as ips


@pytest.mark.parametrize("data_style", ["lst", "single", "attr", "attr_lst", "dct"])
@pytest.mark.parametrize("proj_w_data", [1], indirect=True)
def test_direct_selection(proj_w_data, data_style):
    proj, data = proj_w_data

    with proj:
        if data_style == "lst":
            _data = data
        elif data_style == "single":
            _data = data[0]
        elif data_style == "attr":
            _data = data[0].atoms
        elif data_style == "attr_lst":
            _data = [d.atoms for d in data]
        elif data_style == "dct":
            _data = {"data_0": data[0].atoms}
        else:
            raise ValueError(data_style)

        selection = ips.IndexSelection(data=_data, indices=[0, 1, 2])
        selection_w_exclusion = ips.IndexSelection(
            data=_data,
            indices=[0, 1, 2],
            exclude_configurations=selection.selected_configurations,
            name="selection2",
        )

    proj.repro()
    assert selection.selected_configurations == {"data_0": [0, 1, 2]}
    assert selection_w_exclusion.selected_configurations == {"data_0": [3, 4, 5]}

    assert selection.atoms == data[0].atoms[:3]
    assert selection_w_exclusion.atoms == data[0].atoms[3:6]


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


def test_exclude_configurations(proj_path, traj_file):
    with ips.Project() as project:
        data = ips.AddData(file=traj_file)
        test_data = ips.IndexSelection(
            data=data,
            start=0,
            stop=5,
            step=None,
        )

        train_data = ips.IndexSelection(
            data=data, start=0, stop=5, step=None, exclude=test_data
        )

    project.repro()

    assert train_data.selected_configurations == {"AddData": [5, 6, 7, 8, 9]}
    assert test_data.selected_configurations == {"AddData": [0, 1, 2, 3, 4]}


def test_exclude_configurations_list(proj_path, traj_file):
    train_data = []
    test_data = []
    with ips.Project() as project:
        data = ips.AddData(file=traj_file)
        test_data = ips.IndexSelection(
            data=data.atoms,
            start=0,
            stop=5,
            step=None,
        )
        train_data = ips.IndexSelection(
            data=data.atoms,
            exclude=test_data,
            start=0,
            stop=5,
            step=None,
        )

        validation_data = ips.IndexSelection(
            data=data.atoms,
            exclude=[train_data, test_data],
            start=0,
            stop=5,
            step=None,
        )

    project.repro()

    assert train_data.selected_configurations == {"AddData": [5, 6, 7, 8, 9]}
    assert test_data.selected_configurations == {"AddData": [0, 1, 2, 3, 4]}
    assert validation_data.selected_configurations == {"AddData": [10, 11, 12, 13, 14]}


def test_filter_outlier(proj_path, traj_file):
    with ips.Project() as project:
        data = ips.AddData(file=traj_file)
        filtered_data = ips.FilterOutlier(
            data=data.atoms, key="energy", threshold=1, direction="both"
        )

    project.repro()

    filtered_data = zntrack.from_rev(name=filtered_data.name)

    assert len(filtered_data.atoms) == 13
