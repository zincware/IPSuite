import pathlib
import shutil

import pytest

import ipsuite as ips


@pytest.mark.parametrize(
    "random, max_rattel_shift, cell_fraction",
    [(True, None, [1.0, 1.0]), (False, 0.1, [0.5, 0.5])],
)
def test_rattle_atoms(proj_path, traj_file, random, max_rattel_shift, cell_fraction):
    traj_file = pathlib.Path(traj_file)
    shutil.copy(traj_file, ".")

    n_conf_per_dist = [5, 5]
    z_dist_list = [1, 2]

    with ips.Project() as project:
        data = ips.AddData(file=traj_file.name)

        scan = ips.SurfaceRasterScan(
            data=data.frames,
            symbol="O",
            n_conf_per_dist=n_conf_per_dist,
            z_dist_list=z_dist_list,
            random=random,
            max_rattel_shift=max_rattel_shift,
            cell_fraction=cell_fraction,
            seed=0,
        )
    project.repro()

    data.load()
    scan.load()
    atoms = scan.frames

    desired_num_configs = len(z_dist_list) * n_conf_per_dist[0] * n_conf_per_dist[1]
    assert len(atoms) == desired_num_configs

    # check if distances are equally distributed
    dist1 = atoms[0].positions[-1, 1] - atoms[1].positions[-1, 1]
    dist2 = atoms[2].positions[-1, 1] - atoms[3].positions[-1, 1]
    if random is False:
        assert dist1 == dist2
    else:
        assert dist1 != dist2

    # check if bulk hase still the atom positions of the init structure
    pos1 = data.frames[0].positions[0, 0]
    pos2 = scan.frames[0].positions[0, 0]
    if max_rattel_shift is None:
        assert pos1 == pos2
    else:
        assert pos1 != pos2

    # check if the additive just is added in the correct cell fraction
    pos_additive = []
    for atom in scan.frames:
        pos_additive.append(atom.positions[-1, :2])

    max_x = max(pos_additive[0])
    max_y = max(pos_additive[1])

    cell = data.frames[0].cell

    assert max_x * cell_fraction[0] <= cell[0, 0]
    assert max_y * cell_fraction[1] <= cell[1, 1]
