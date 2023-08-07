import typing

import ase.geometry
import ase.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import zntrack

from ipsuite import base


def nonuniform_imshow(ax, x, y, z, aspect=1, cmap=plt.cm.rainbow):
    """Plot a non-uniformly sampled 2D array.

    References
    ----------
    adapted from https://stackoverflow.com/a/53780594/10504481
    """
    # Create regular grid
    xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate missing data
    rbf = scipy.interpolate.Rbf(x, y, z, function="linear")
    zi = rbf(xi, yi)

    _ = ax.imshow(
        zi,
        interpolation="nearest",
        cmap=cmap,
        extent=[x.min(), x.max(), y.max(), y.min()],
    )
    # ax.scatter(x, y, marker="x", s=2.5)
    ax.set_aspect(aspect)


class MoveSingleParticle(base.IPSNode):
    """Move a single particle in a given direction."""

    atoms_list = zntrack.zn.deps()
    atoms_list_id = zntrack.zn.params(0)  # the atoms object in the atoms list
    atom_id = zntrack.zn.params(0)  # the atom id to move
    scale = zntrack.zn.params(0.5)  # the standard deviation of the normal distribution
    seed = zntrack.zn.params(1234)

    samples = zntrack.zn.params(10)  # how many samples to take

    atoms: list = zntrack.zn.outs()
    atoms_path = zntrack.dvc.outs(zntrack.nwd / "atoms")

    def run(self):
        """ZnTrack run method."""
        self.atoms = []
        np.random.seed(self.seed)
        self.atoms_path.mkdir(parents=True, exist_ok=True)
        for idx in range(self.samples):
            atoms = self.atoms_list[self.atoms_list_id].copy()
            atoms.positions[self.atom_id] += np.random.normal(scale=self.scale, size=3)
            self.atoms.append(atoms)
            ase.io.write(self.atoms_path / f"atoms_{idx}.xyz", atoms)

    def get_atom_filenames(self):
        return [str(self.atoms_path / f"atoms_{idx}.xyz") for idx in range(self.samples)]


class AnalyseGlobalForceSensitivity(base.IPSNode):
    atoms_list = zntrack.zn.deps()
    plots = zntrack.dvc.outs(zntrack.nwd / "plots")

    def run(self):
        # assume all atoms have only a single particle changed
        r_ij, d_ij = ase.geometry.get_distances(self.atoms_list[-1].positions)

        forces = np.stack([atoms.get_forces() for atoms in self.atoms_list])
        std_forces = np.std(forces, axis=0)
        mean_forces = np.sum(std_forces, axis=1)

        fig, ax = plt.subplots()
        nonuniform_imshow(ax, r_ij[0, :, 0], r_ij[0, :, 1], mean_forces)
        fig.savefig(self.plots / "2d_forces.png")

        fig, ax = plt.subplots()
        ax.scatter(d_ij[0], np.sum(std_forces, axis=1))
        ax.set_yscale("log")
        ax.set_xlabel(r"distance $d ~ / ~ \AA$")
        ax.set_ylabel(r"standard deviation $\sigma ~ / ~ a.u.$")
        fig.savefig(self.plots / "std_forces.png")


def _compute_std_leave_one_out(data):  # Leave-One-Out Cross-Validation
    std = [
        np.std([val for k, val in enumerate(data) if k != idx])
        for idx in range(len(data))
    ]
    return np.std(data), scipy.stats.sem(std)


class IsConstraintMD(typing.Protocol):
    """Protocol for objects that have a results attribute."""

    selected_atom_id: int
    radius: float


class AnalyseSingleForceSensitivity(zntrack.Node):
    data: list[list[ase.Atoms]] = zntrack.zn.deps()
    sim_list: list = zntrack.zn.deps()  # list["ASEMD"]

    alpha: float = zntrack.zn.params(
        0.05
    )  # Desired significance level (e.g., 95% confidence interval)

    sensitivity: pd.DataFrame = zntrack.zn.plots()
    sensitivity_plot: str = zntrack.dvc.outs(zntrack.nwd / "sensitivity.png")

    def t_confidence_interval(self, data):
        """Returns the confidence interval for the given data and significance level."""
        df = len(data) - 1

        sample_variance = np.var(data, ddof=1)
        # lower_chi2 = stats.chi2.ppf(alpha / 2, df)
        upper_chi2 = scipy.stats.chi2.ppf(1 - self.alpha / 2, df)

        lower_bound = np.sqrt((df * sample_variance) / upper_chi2)
        # upper_bound = np.sqrt((df * sample_variance) / lower_chi2)

        return np.std(data) - lower_bound

    def get_values(self, data, item):
        forces = np.array([x.get_forces() for x in data])
        val = np.linalg.norm(forces[:, item], axis=1)
        return np.std(val, ddof=1), self.t_confidence_interval(val)

    def run(self):
        values = []

        self.data = [x.atoms if isinstance(x, zntrack.Node) else x for x in self.data]

        for atoms, sim in zip(self.data, self.sim_list):
            radius = sim.constraint_list[0].radius
            atom_id = sim.constraint_list[0].get_selected_atom_id(atoms[0])

            value = self.get_values(atoms, atom_id)
            values.append({"radius": radius, "std": value[0], "ci": value[1]})

        self.sensitivity = pd.DataFrame(values).set_index("radius").sort_index()

        fig, ax = plt.subplots()
        ax.errorbar(
            x=self.sensitivity.index,
            y=self.sensitivity["std"],
            yerr=self.sensitivity["ci"],
            capsize=5,
        )
        ax.set_ylabel(r"Standard deviation $\sigma$ of force $f$")
        ax.set_xlabel(r"Distance $r ~ / ~ \AA$")
        ax.set_yscale("log")
        fig.savefig(self.sensitivity_plot, bbox_inches="tight")
