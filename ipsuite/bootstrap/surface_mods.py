import logging

import ase
import matplotlib.pyplot as plt
import numpy as np
import zntrack
from ase.cell import Cell
from numpy.random import default_rng

from ipsuite import analysis, base

log = logging.getLogger(__name__)


class SurfaceRasterScan(base.ProcessSingleAtom):
    symbol: int = zntrack.params()
    z_dist_list: list[int] = zntrack.params()
    n_conf_per_dist: list[int] = zntrack.params([5, 5])
    cell_fraction: list[float] = zntrack.params([1, 1])
    random: bool = zntrack.params(False)
    rattel: bool = zntrack.params(False)
    max_shift: float = zntrack.params(None)
    seed: bool = zntrack.params(1)

    def run(self) -> None:
        rng = default_rng(self.seed)

        atoms = self.get_data()

        cell = atoms.cell
        cellpar = cell.cellpar()
        cell = np.array(cell)

        z_max = max(atoms.get_positions()[:, 2])

        if not isinstance(self.n_conf_per_dist, list):
            self.n_conf_per_dist = [self.n_conf_per_dist, self.n_conf_per_dist]
        if not isinstance(self.cell_fraction, list):
            self.cell_fraction = [self.cell_fraction, self.cell_fraction]
        atoms_list = []
        for z_dist in self.z_dist_list:
            if cellpar[2] < z_max + z_dist + 10:
                cellpar[2] = z_max + z_dist + 10
                new_cell = Cell.fromcellpar(cellpar)
                atoms.set_cell(new_cell)
                log.warning("vacuum was extended")

            if not self.random:
                a_scaling = np.linspace(0, 1, self.n_conf_per_dist[0])
                b_scaling = np.linspace(0, 1, self.n_conf_per_dist[1])
            else:
                a_scaling = np.random.rand(self.n_conf_per_dist[0])
                a_scaling = np.sort(a_scaling)
                b_scaling = np.random.rand(self.n_conf_per_dist[1])
                b_scaling = np.sort(b_scaling)

            a_vec = cell[0, :2] * self.cell_fraction[0]
            scaled_a_vecs = a_scaling[:, np.newaxis] * a_vec
            b_vec = cell[1, :2] * self.cell_fraction[1]
            scaled_b_vecs = b_scaling[:, np.newaxis] * b_vec

            if self.rattel and self.max_shift is None:
                raise ValueError("max_shift must be set.")

            for a in scaled_a_vecs:
                for b in scaled_b_vecs:
                    if self.rattel:
                        new_atoms = atoms.copy()
                        displacement = rng.uniform(
                            -self.max_shift,
                            self.max_shift,
                            size=new_atoms.positions.shape,
                        )
                        new_atoms.positions += displacement
                        atoms_list.append(new_atoms)
                    else:
                        atoms_list.append(atoms.copy())

                    cart_pos = a + b
                    extension = ase.Atoms(
                        self.symbol, [[cart_pos[0], cart_pos[1], z_max + z_dist]]
                    )
                    atoms_list[-1].extend(extension)

        self.atoms = atoms_list


class SurfaceRasterMetrics(analysis.PredictionMetrics):
    scan_node: SurfaceRasterScan = zntrack.deps()
    seed: int = zntrack.params(0)

    def get_plots(self, save=False):
        self.plots_dir.mkdir(exist_ok=True)

        pos = []
        for atoms in self.data.atoms:
            pos.append(atoms.positions[-1])
        pos = np.array(pos)

        shape = self.scan_node.n_conf_per_dist
        n_distances = len(self.scan_node.z_dist_list)
        shape.append(n_distances)

        x_pos = np.reshape(pos[:, 0], shape)
        for j in range(x_pos.shape[1]):
            x_pos[j, :] = x_pos[j, 0]

        # has to be removed
        x_pos_cache = x_pos
        x_pos = x_pos[3:]
        x_pos = np.insert(x_pos, 4, x_pos_cache[0, :], axis=0)
        x_pos = np.insert(x_pos, 6, x_pos_cache[1, :], axis=0)
        x_pos = np.insert(x_pos, 8, x_pos_cache[2, :], axis=0)
        ###################

        y_pos = np.reshape(pos[:, 1], shape)
        t_E = np.reshape(self.energy_df["true"], shape)
        p_E = np.reshape(self.energy_df["prediction"], shape)

        for distance in self.scan_node.z_dist_list:
            plot_heat(x_pos, y_pos, t_E, "E_abinitio", distance, plots_dir=self.plots_dir)
            plot_heat(
                x_pos, y_pos, p_E, "E_predicted", distance, plots_dir=self.plots_dir
            )
            plot_heat_both(
                x_pos,
                y_pos,
                [t_E, p_E],
                "E_predicted",
                distance,
                plots_dir=self.plots_dir,
            )


def plot_heat(x, y, z, name, height, plots_dir):
    fig, ax = plt.subplots(layout="constrained")
    cm = ax.pcolormesh(x, y, z)
    
    ax.axis("scaled")
    ax.set_title(f"{name} for additive at {height} ang dist to surface")
    ax.set_xlabel("x-pos additiv [ang]")
    ax.set_ylabel("y-pos additiv [ang]")
    cbar = fig.colorbar(cm)
    cbar.ax.set_ylabel(f"{name}")
    fig.savefig(plots_dir / f"{name}-{height}-heat.png")


def plot_heat_both(x, y, z, name, height, plots_dir):
    fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(8, 3.5))
    for i, ax in enumerate(axes.flat):
        cm = ax.pcolormesh(x, y, z[i])
        ax.axis("scaled")
        ax.set_xlabel("x-pos additiv [ang]")
        ax.set_ylabel("y-pos additiv [ang]")
    axes[0].set_title(f"true-{name}")
    axes[1].set_title(f"predicted-{name}")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.015, 0.03, 0.87])
    fig.colorbar(cm, cax=cbar_ax)

    if name == "energy":
        cbar_ax.set_ylabel("Energy [meV/atom]")
    if name == "force":
        cbar_ax.set_ylabel("Magnetude of force per atom [meV/ang]")

    fig.suptitle(f"Additive {height} ang over the surface")
    fig.savefig(plots_dir / f"{name}-{height}-heat.png")


def get_pos_and_metrics(node, atom_num):
    pos = []
    energy = []
    forces = []
    for atoms in node.atoms:
        pos.append(atoms.positions[atom_num])
        energy.append(atoms.get_potential_energy())
        forces.append(atoms.get_forces()[atom_num, :])
    return np.array(pos), np.array(energy), np.array(forces)
