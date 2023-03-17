import random

import ase.io
import pytest
import tqdm

import ipsuite


@pytest.mark.parametrize("param", ("list+at", "at"))  #  "list", "single", "with_ids"
def test_train_validation_selection(proj_path, traj_file, param):
    with ipsuite.Project() as project:
        data_1 = ipsuite.AddData(file=traj_file, name="data_1")
        # different test cases
        train_tests = {
            "list+at": [data_1.atoms],
            "at": data_1.atoms,
            "list": [data_1],
            "single": data_1,
            # only affects exclude_configurations
            "with_ids": data_1,
        }

        train_selection = ipsuite.configuration_selection.RandomSelection(
            data=train_tests[param], n_configurations=10, name="train_data"
        )

        # different test cases
        validation_tests = {
            # limit the data directly
            "list+at": {"data": [train_selection.excluded_atoms]},
            "at": {"data": train_selection.excluded_atoms},
            # test with exclude_configurations
            "list": {"data": data_1, "exclude_configurations": [train_selection]},
            "single": {"data": data_1, "exclude_configurations": train_selection},
            "with_ids": {
                "data": data_1,
                "exclude_configurations": train_selection.selected_configurations,
            },
        }

        validation_selection = ipsuite.configuration_selection.RandomSelection(
            **validation_tests[param], n_configurations=8, name="val_data"
        )
    project.run()

    train_selection.load()
    validation_selection.load()

    assert isinstance(validation_selection.atoms[0], ase.Atoms)
    assert isinstance(train_selection.atoms[0], ase.Atoms)
    assert len(train_selection.atoms) == 10
    assert len(validation_selection.atoms) == 8

    for train_atom in tqdm.tqdm(train_selection.atoms):
        assert train_atom not in validation_selection.atoms[:]


@pytest.mark.parametrize("eager", [False])
def test_KernelSelect(proj_path, traj_file, eager):
    mmk_kernel = ipsuite.configuration_comparison.MMKernel(
        use_jit=True,
        soap={
            "atomic_keys": [6, 8],
            "r_cut": 1.1,
            "n_max": 3,
            "l_max": 3,
            "sigma": 0.5,
        },
    )

    rematch_kernel = ipsuite.configuration_comparison.REMatch(
        soap={
            "atomic_keys": [6, 8],
            "r_cut": 1.1,
            "n_max": 3,
            "l_max": 3,
            "sigma": 0.5,
        }
    )

    with ipsuite.Project() as project:
        data_1 = ipsuite.AddData(file=traj_file, name="data_1")
        seed_configs = ipsuite.configuration_selection.RandomSelection(
            data=data_1, n_configurations=1, seed=42, name="seed"
        )
        mmk_selection = ipsuite.configuration_selection.KernelSelectionNode(
            correlation_time=1,
            n_configurations=5,
            kernel=mmk_kernel,
            initial_configurations=seed_configs.atoms,
            data=data_1,
            name="MMK",
        )
        REMatch_selection = ipsuite.configuration_selection.KernelSelectionNode(
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
    selection = ipsuite.configuration_selection.RandomSelection(
        data=None, data_file=traj_file, n_configurations=5
    )
    selection.update_data()
    assert isinstance(selection.data, list)
    assert isinstance(selection.data[0], ase.Atoms)
    with ipsuite.Project() as proj:
        selection = ipsuite.configuration_selection.RandomSelection(
            data=None, data_file=traj_file, n_configurations=5
        )
    proj.run(eager=eager)
    if not eager:
        selection.load()
    assert len(selection.atoms) == 5
    assert isinstance(selection.atoms[0], ase.Atoms)


@pytest.fixture
def test_traj(tmp_path_factory) -> (str, int):
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


def __test_MMKSelectMethod(proj_path, test_traj):
    project = ipsuite.Project()

    test_traj, n_configurations = test_traj

    data = ipsuite.AddData(file=test_traj)
    seed_configs = ipsuite.configuration_selection.RandomSelection(
        data=data, n_configurations=1, name="seed"
    )
    mmk_selection = ipsuite.configuration_selection.KernelSelectionNode(
        correlation_time=1,
        n_configurations=n_configurations - 1,  # remove the seed configuration
        kernel=ipsuite.configuration_comparison.MMKernel(
            use_jit=False,
            soap={
                "atomic_keys": [1, 6],
                "r_cut": 3,
                "n_max": 3,
                "l_max": 3,
                "sigma": 0.5,
            },
        ),
        initial_configurations=seed_configs @ "atoms",
        data=data,
        name="MMK",
    )
    rematch_selection = ipsuite.configuration_selection.KernelSelectionNode(
        correlation_time=1,
        n_configurations=n_configurations - 1,  # remove the seed configuration
        kernel=ipsuite.configuration_comparison.REMatch(
            soap={
                "atomic_keys": [1, 6],
                "r_cut": 3,
                "n_max": 3,
                "l_max": 3,
                "sigma": 0.5,
            }
        ),
        initial_configurations=seed_configs @ "atoms",
        data=data,
        name="REMatch",
    )

    nodes = [data, seed_configs, mmk_selection, rematch_selection]
    project.add(nodes)  # write graph
    project.repro()

    loaded_mmk_selection = ipsuite.configuration_selection.KernelSelectionNode.load("MMK")
    loaded_REMatch_selection = ipsuite.configuration_selection.KernelSelectionNode.load(
        "REMatch"
    )

    assert len(loaded_mmk_selection.atoms) == n_configurations - 1
    assert isinstance(loaded_mmk_selection.atoms[0], ase.Atoms)
    assert len(loaded_REMatch_selection.atoms) == n_configurations - 1
    assert isinstance(loaded_REMatch_selection.atoms[0], ase.Atoms)

    confs_mmk = loaded_mmk_selection.selected_configurations["AddData"]
    confs_REMatch = loaded_REMatch_selection.selected_configurations["AddData"]

    # need to add the seed configuration
    confs_mmk += ipsuite.configuration_selection.RandomSelection[
        "seed"
    ].selected_configurations["AddData"]

    confs_mmk = [x if x < n_configurations else x - n_configurations for x in confs_mmk]
    confs_REMatch += ipsuite.configuration_selection.RandomSelection[
        "seed"
    ].selected_configurations["AddData"]

    confs_REMatch = [
        x if x < n_configurations else x - n_configurations for x in confs_REMatch
    ]
    assert set(confs_mmk) == set(range(n_configurations))
    assert set(confs_REMatch) == set(range(n_configurations))
