import random
import typing

import ase
import pytest

import ipsuite as ips


@pytest.mark.parametrize(
    ("cls", "selected_ids"),
    [
        (ips.configuration_selection.RandomSelection, [2, 3, 13]),
        (ips.configuration_selection.UniformEnergeticSelection, [0, 10, 20]),
        # they are the same because energy is increasing uniformly
        (ips.configuration_selection.UniformTemporalSelection, [0, 10, 20]),
    ],
)
@pytest.mark.parametrize("eager", [True, False])
def test_configuration_selection(proj_path, traj_file, eager, cls, selected_ids):
    with ips.Project() as project:
        data = ips.AddData(file=traj_file)
        selection = cls(data=data.atoms, n_configurations=3)

    project.run(eager=eager, save=not eager)

    if not eager:
        selection.load()
        data.load()

    assert selection.atoms == [data.atoms[x] for x in selected_ids]


@pytest.mark.parametrize("eager", [True, False])
def test_UniformArangeSelection(proj_path, traj_file, eager):
    with ips.Project() as project:
        data = [
            ips.AddData(file=traj_file, name="data1"),
            ips.AddData(file=traj_file, name="data2"),
        ]
        selection = ips.configuration_selection.UniformArangeSelection(data=data, step=10)

    project.run(eager=eager, save=not eager)

    if not eager:
        selection.load()

    assert selection.selected_configurations == {"data1": [0, 10, 20], "data2": [9, 19]}


@pytest.mark.parametrize("eager", [True, False])
def test_SplitSelection(proj_path, traj_file, eager):
    with ips.Project() as project:
        data = [
            ips.AddData(file=traj_file, name="data1"),
            ips.AddData(file=traj_file, name="data2"),
        ]
        selection = ips.configuration_selection.SplitSelection(data=data, split=0.3)

    project.run(eager=eager, save=not eager)

    if not eager:
        selection.load()

    assert selection.selected_configurations == {"data1": list(range(12)), "data2": []}


@pytest.mark.parametrize("eager", [False])
def test_KernelSelect(proj_path, traj_file, eager):
    mmk_kernel = ips.configuration_comparison.MMKernel(
        use_jit=True,
        soap={
            "r_cut": 1.1,
            "n_max": 3,
            "l_max": 3,
            "sigma": 0.5,
        },
    )

    rematch_kernel = ips.configuration_comparison.REMatch(
        soap={
            "r_cut": 1.1,
            "n_max": 3,
            "l_max": 3,
            "sigma": 0.5,
        }
    )

    with ips.Project() as project:
        data_1 = ips.AddData(file=traj_file, name="data_1")
        seed_configs = ips.configuration_selection.RandomSelection(
            data=data_1, n_configurations=1, seed=42, name="seed"
        )
        mmk_selection = ips.configuration_selection.KernelSelection(
            correlation_time=1,
            n_configurations=5,
            kernel=mmk_kernel,
            initial_configurations=seed_configs.atoms,
            data=data_1,
            name="MMK",
        )
        REMatch_selection = ips.configuration_selection.KernelSelection(
            correlation_time=1,
            n_configurations=5,
            kernel=rematch_kernel,
            initial_configurations=seed_configs.atoms,
            data=data_1,
            name="REMatch",
        )

    project.run(eager=eager)
    if not eager:
        mmk_selection.load()
        REMatch_selection.load()

    assert len(mmk_selection.atoms) == 5
    assert isinstance(mmk_selection.atoms[0], ase.Atoms)
    assert len(REMatch_selection.atoms) == 5
    assert isinstance(REMatch_selection.atoms[0], ase.Atoms)


@pytest.mark.parametrize("eager", [True, False])
def test_select_from_file(proj_path, traj_file, eager):
    with ips.Project() as proj:
        selection = ips.configuration_selection.RandomSelection(
            data=None, data_file=traj_file, n_configurations=5
        )
    proj.run(eager=eager)
    if not eager:
        selection.load()
    assert len(selection.atoms) == 5
    assert isinstance(selection.atoms[0], ase.Atoms)


@pytest.fixture
def test_traj(tmp_path_factory) -> typing.Tuple[str, int]:
    """Generate n atoms objects. The first n/2 are random shifts of a CH4 tetraeder.
    The last n/2 are a copy of these with a slightly smaller shift.

    The MMK should select every randomly shifted configuration only once. Either with
    or without the additional random shift.
    """
    tetraeder = ase.Atoms(
        "CH4",
        positions=[(1, 1, 1), (0, 0, 0), (0, 2, 2), (2, 2, 0), (2, 0, 2)],
        cell=(2, 2, 2),
    )

    random.seed(42)

    n_configurations = 20  # it will return twice as many configurations

    atoms = [tetraeder.copy() for _ in range(n_configurations)]
    [x.rattle(stdev=0.5, seed=random.randint(1, int(1e6))) for x in atoms]
    # create a replicate
    atoms_replicated = [x.copy() for x in atoms]
    [x.rattle(stdev=0.05, seed=random.randint(1, int(1e6))) for x in atoms_replicated]

    atoms += atoms_replicated

    temporary_path = tmp_path_factory.getbasetemp()
    file = temporary_path / "mmk_test.extxyz"
    ase.io.write(file, atoms)
    return file.as_posix(), n_configurations


def test_MMKSelectMethod(proj_path, test_traj):
    test_traj, n_configurations = test_traj

    rematch_kernel = ips.configuration_comparison.REMatch(
        soap={
            "r_cut": 3,
            "n_max": 3,
            "l_max": 3,
            "sigma": 0.5,
        }
    )
    mmk_kernel = ips.configuration_comparison.MMKernel(
        # use_jit=False,
        soap={
            "r_cut": 3,
            "n_max": 3,
            "l_max": 3,
            "sigma": 0.5,
        },
    )

    with ips.Project() as project:
        data = ips.AddData(file=test_traj)
        seed_configs = ips.configuration_selection.RandomSelection(
            data=data, n_configurations=1, name="seed"
        )
        mmk_selection = ips.configuration_selection.KernelSelection(
            correlation_time=1,
            n_configurations=n_configurations - 1,  # remove the seed configuration
            kernel=mmk_kernel,
            initial_configurations=seed_configs.atoms,
            data=data,
            name="MMK",
        )
        rematch_selection = ips.configuration_selection.KernelSelection(
            correlation_time=1,
            n_configurations=n_configurations - 1,  # remove the seed configuration
            kernel=rematch_kernel,
            initial_configurations=seed_configs.atoms,
            data=data,
            name="REMatch",
        )

    project.run()

    mmk_selection.load()
    rematch_selection.load()

    assert len(mmk_selection.atoms) == n_configurations - 1
    assert isinstance(mmk_selection.atoms[0], ase.Atoms)
    assert len(rematch_selection.atoms) == n_configurations - 1
    assert isinstance(rematch_selection.atoms[0], ase.Atoms)

    confs_mmk = mmk_selection.selected_configurations["AddData"]
    confs_REMatch = rematch_selection.selected_configurations["AddData"]

    # need to add the seed configuration
    seed_configs.load()
    confs_mmk += seed_configs.selected_configurations["AddData"]

    confs_mmk = [x if x < n_configurations else x - n_configurations for x in confs_mmk]
    confs_REMatch += seed_configs.selected_configurations["AddData"]

    confs_REMatch = [
        x if x < n_configurations else x - n_configurations for x in confs_REMatch
    ]
    assert set(confs_mmk) == set(range(n_configurations))
    assert set(confs_REMatch) == set(range(n_configurations))
