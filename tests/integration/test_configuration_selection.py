import ipsuite as ips
import pytest
import ase


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
        data[0].load()
        data[1].load()

    atoms = data[0].atoms + data[1].atoms

    assert selection.selected_configurations == {"data1": [0, 10, 20], "data2": [9, 19]}


@pytest.mark.parametrize("eager", [False])
def test_KernelSelect(proj_path, traj_file, eager):
    mmk_kernel = ips.configuration_comparison.MMKernel(
        use_jit=True,
        soap={
            "atomic_keys": [6, 8],
            "r_cut": 1.1,
            "n_max": 3,
            "l_max": 3,
            "sigma": 0.5,
        },
    )

    rematch_kernel = ips.configuration_comparison.REMatch(
        soap={
            "atomic_keys": [6, 8],
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
        mmk_selection = ips.configuration_selection.KernelSelectionNode(
            correlation_time=1,
            n_configurations=5,
            kernel=mmk_kernel,
            initial_configurations=seed_configs.atoms,
            data=data_1,
            name="MMK",
        )
        REMatch_selection = ips.configuration_selection.KernelSelectionNode(
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
