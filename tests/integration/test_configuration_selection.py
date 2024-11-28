import random
import typing

import ase
import pytest
import zntrack

import ipsuite as ips


@pytest.mark.parametrize(
    ("cls", "selected_ids"),
    [
        (ips.RandomSelection, [2, 3, 13]),
        (ips.UniformEnergeticSelection, [0, 10, 20]),
        # # they are the same because energy is increasing uniformly
        (ips.UniformTemporalSelection, [0, 10, 20]),
    ],
)
def test_configuration_selection(proj_path, traj_file, cls, selected_ids):
    with ips.Project() as project:
        data = ips.AddData(file=traj_file)
        selection = cls(data=data.atoms, n_configurations=3)

    project.repro()

    selection = zntrack.from_rev(name=selection.name)
    assert selection.atoms == [data.atoms[x] for x in selected_ids]


def test_UniformArangeSelection(proj_path, traj_file):
    with ips.Project() as project:
        data = [
            ips.AddData(file=traj_file, name="data1").atoms,
            ips.AddData(file=traj_file, name="data2").atoms,
        ]
        selection = ips.UniformArangeSelection(data=sum(data, []), step=10)

    project.repro()

    assert selection.selected_ids == [0, 10, 20, 30, 40]


def test_SplitSelection(proj_path, traj_file):
    with ips.Project() as project:
        data = [
            ips.AddData(file=traj_file, name="data1").atoms,
            ips.AddData(file=traj_file, name="data2").atoms,
        ]
        selection = ips.SplitSelection(data=sum(data, []), split=0.3)

    project.repro()

    assert selection.selected_ids == list(range(12))
