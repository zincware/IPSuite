import contextlib
import logging
import pathlib
import typing
from collections import deque

import ase
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
import znh5md
import zntrack
from ase import units
from ase.calculators.calculator import PropertyNotImplementedError
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.neighborlist import build_neighbor_list
from numpy.random import default_rng
from scipy.interpolate import interpn
from tqdm import trange

from ipsuite import base, models, utils
from ipsuite.analysis.bin_property import get_histogram_figure
from ipsuite.geometry import BarycenterMapping
from ipsuite.utils.md import get_energy_terms

log = logging.getLogger(__name__)


class PredictWithModel(base.ProcessAtoms):
    """Create and Save the predictions from model on atoms.

    Attributes
    ----------
    model: The MLModel node that implements the 'predict' method
    atoms: list[Atoms] to predict properties for

    predictions: list[Atoms] the atoms that have the predicted properties from model
    """

    model: models.MLModel = zntrack.zn.deps()

    def run(self):
        self.atoms = []
        calc = self.model.calc
        for configuration in tqdm.tqdm(self.get_data(), ncols=70):
            configuration: ase.Atoms
            # Run calculation
            atoms = configuration.copy()
            atoms.calc = calc

            # Save properties to SinglePointCalculator
            # (Other calculators might not be saved.)
            properties = {}
            with contextlib.suppress(
                ase.calculators.singlepoint.PropertyNotImplementedError
            ):
                properties["energy"] = atoms.get_potential_energy()
            with contextlib.suppress(
                ase.calculators.singlepoint.PropertyNotImplementedError
            ):
                properties["forces"] = atoms.get_forces()
            with contextlib.suppress(
                ase.calculators.singlepoint.PropertyNotImplementedError
            ):
                properties["stress"] = atoms.get_stress()

            if properties:
                atoms.calc = ase.calculators.singlepoint.SinglePointCalculator(
                    atoms=atoms, **properties
                )
            self.atoms.append(atoms)


def density_scatter(ax, x, y, bins, **kwargs) -> None:
    """Create a scatter plot colored by 2d histogram density.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
    x: np.ndarray
    y: np.ndarray
    bins: int
    kwargs
        any kwargs passed to 'ax.scatter'

    Returns
    -------

    References
    ----------
    Adapted from https://stackoverflow.com/a/53865762/10504481

    """
    # convert e.g. DataFrame to numpy array values
    x = np.array(x)
    y = np.array(y)

    if "cmap" not in kwargs:
        kwargs["cmap"] = "viridis"

    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    points = (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1]))
    xi = np.vstack([x, y]).T
    z = interpn(points, data, xi, method="splinef2d", bounds_error=False)
    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    ax.scatter(x, y, c=z, **kwargs)


def get_figure(
    true, prediction, datalabel: str, xlabel: str, ylabel: str, figsize: tuple = (10, 7)
) -> plt.Figure:
    """Create a correlation plot for true, prediction values.

    Parameters
    ----------
    true: the true values
    prediction: the predicted values
    datalabel: str, the label for the prediction, e.g. 'MAE: 0.123 meV'
    xlabel: str, the xlabel
    ylabel: str, the xlabel
    figsize: tuple, size of the figure

    Returns
    -------
    plt.Figure

    """
    sns.set()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(true, true, color="grey", zorder=0)  # plot the diagonal in the background
    bins = 500 if (len(true) / 10) > 500 else int(len(true) * 0.1)
    if bins < 20:
        # don't use density for very small datasets
        ax.scatter(true, prediction, marker="x", s=20.0, label=datalabel)
    else:
        density_scatter(
            ax, true, prediction, bins=bins, marker="x", s=20.0, label=datalabel
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig


class AnalysePrediction(base.AnalyseProcessAtoms):
    """Analyse the Models Prediction.

    This Node computes
    - MAE
    - RMSE
    - L4 Error
    - Maxium Error
    """

    energy_df_file: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "energy_df.csv")
    forces_df_file: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "forces_df.csv")
    stress_df_file: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "stress_df.csv")

    energy: dict = zntrack.zn.metrics()
    forces: dict = zntrack.zn.metrics()
    stress: dict = zntrack.zn.metrics()

    plots_dir: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "plots")

    energy_df: pd.DataFrame
    forces_df: pd.DataFrame
    stress_df: pd.DataFrame

    def _post_load_(self):
        """Load metrics - if available."""
        try:
            self.energy_df = pd.read_csv(self.energy_df_file)
        except FileNotFoundError:
            self.energy_df = pd.DataFrame({})
        try:
            self.forces_df = pd.read_csv(self.forces_df_file)
        except FileNotFoundError:
            self.forces_df = pd.DataFrame({})
        try:
            self.stress_df = pd.read_csv(self.stress_df_file)
        except FileNotFoundError:
            self.stress_df = pd.DataFrame({})

    def get_dataframes(self):
        """Create a pandas dataframe from the given data."""
        true_data, pred_data = self.get_data()

        self.energy_df = pd.DataFrame(
            {
                "true": [x.get_potential_energy() for x in true_data],
                "prediction": [x.get_potential_energy() for x in pred_data],
            }
        )

        try:
            true_forces = np.reshape([x.get_forces() for x in true_data], (-1, 3))
            pred_forces = np.reshape([x.get_forces() for x in pred_data], (-1, 3))

            self.forces_df = pd.DataFrame(
                {
                    "true": np.linalg.norm(true_forces, axis=-1),
                    "true_x": true_forces[:, 0],
                    "true_y": true_forces[:, 1],
                    "true_z": true_forces[:, 2],
                    "prediction": np.linalg.norm(pred_forces, axis=-1),
                    "prediction_x": pred_forces[:, 0],
                    "prediction_y": pred_forces[:, 1],
                    "prediction_z": pred_forces[:, 2],
                }
            )
        except PropertyNotImplementedError:
            self.forces_df = pd.DataFrame({})

        try:
            true_stress = np.reshape([x.get_stress() for x in true_data], -1)
            pred_stress = np.reshape([x.get_stress() for x in pred_data], -1)

            self.stress_df = pd.DataFrame(
                {
                    "true": true_stress,
                    "prediction": pred_stress,
                }
            )
        except PropertyNotImplementedError:
            self.stress_df = pd.DataFrame({})

    def get_metrics(self):
        """Update the metrics."""
        self.energy = utils.metrics.get_full_metrics(
            np.array(self.energy_df["true"]), np.array(self.energy_df["prediction"])
        )

        if not self.forces_df.empty:
            self.forces = utils.metrics.get_full_metrics(
                np.array(self.forces_df["true"]), np.array(self.forces_df["prediction"])
            )
        else:
            self.forces = {}

        if not self.stress_df.empty:
            self.stress = utils.metrics.get_full_metrics(
                np.array(self.stress_df["true"]), np.array(self.stress_df["prediction"])
            )
        else:
            self.stress = {}

    def get_plots(self, save=False):
        """Create figures for all available data."""
        self.plots_dir.mkdir(exist_ok=True)

        energy_plot = get_figure(
            self.energy_df["true"],
            self.energy_df["prediction"],
            datalabel=f"MAE: {self.energy['mae']:.4f} meV/atom",
            xlabel=r"$ab~initio$ energy $E$ / eV",
            ylabel=r"predicted energy $E$ / eV",
        )
        if save:
            energy_plot.savefig(self.plots_dir / "energy.png")

        if not self.forces_df.empty:
            forces_plot = get_figure(
                self.forces_df["true"],
                self.forces_df["prediction"],
                datalabel=rf"MAE: {self.forces['mae']:.4f} meV$ / (\AA \cdot $atom)",
                xlabel=(
                    r"$ab~initio$ magnitude of force per atom $|F|$ / eV$ \cdot \AA^{-1}$"
                ),
                ylabel=(
                    r"predicted magnitude of force per atom $|F|$ / eV$ \cdot \AA^{-1}$"
                ),
            )
            if save:
                forces_plot.savefig(self.plots_dir / "forces.png")

        if not self.stress_df.empty:
            stress_plot = get_figure(
                self.stress_df["true"],
                self.stress_df["prediction"],
                datalabel=rf"Max: {self.stress['max']:.4f}",
                xlabel=r"$ab~initio$ stress",
                ylabel=r"predicted stress",
            )
            if save:
                stress_plot.savefig(self.plots_dir / "stress.png")

    def run(self):
        self.nwd.mkdir(exist_ok=True, parents=True)
        self.get_dataframes()
        self.get_metrics()
        self.get_plots(save=True)

        self.energy_df.to_csv(self.energy_df_file)
        self.forces_df.to_csv(self.forces_df_file)
        self.stress_df.to_csv(self.stress_df_file)


class AnalyseForceAngles(base.AnalyseProcessAtoms):
    plot: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "angle.png")
    log_plot: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "angle_ylog.png")

    angles: dict = zntrack.zn.metrics()

    def run(self):
        true_data, pred_data = self.get_data()
        true_forces = np.reshape([x.get_forces() for x in true_data], (-1, 3))
        pred_forces = np.reshape([x.get_forces() for x in pred_data], (-1, 3))

        angles = utils.metrics.get_angles(true_forces, pred_forces)

        self.angles = {
            "rmse": utils.metrics.calculate_l_p_norm(np.zeros_like(angles), angles, p=2),
            "lp4": utils.metrics.calculate_l_p_norm(np.zeros_like(angles), angles, p=4),
            "max": utils.metrics.maximum_error(np.zeros_like(angles), angles),
            "mae": utils.metrics.calculate_l_p_norm(np.zeros_like(angles), angles, p=1),
        }

        fig, ax = plt.subplots()

        sns.histplot(
            angles,
            ax=ax,
            stat="percent",
            label=rf"MAE: ${self.angles['mae']:.2f}^\circ$",
        )
        ax.set_xlabel(r"Angle between true and predicted forces $\theta / ^\circ$ ")
        ax.set_ylabel("Probability / %")
        ax.legend()
        fig.savefig(self.plot)
        ax.set_yscale("log")
        fig.savefig(self.log_plot)


class RattleAnalysis(base.ProcessSingleAtom):
    """Move particles with a given stdev from a starting configuration and predict.

    Attributes
    ----------
    model: The MLModel node that implements the 'predict' method
    atoms: list[Atoms] to predict properties for
    logspace: bool, default=True
        Increase the stdev of rattle with 'np.logspace' instead of 'np.linspace'
    stop: float, default = 1.0
        The stop value for the generated space of stdev points
    num: int, default = 100
        The size of the generated space of stdev points
    factor: float, default = 0.001
        The 'np.linspace(0.0, stop, num) * factor'
    atom_id: int, default = 0
        The atom to pick from self.atoms as a starting point
    """

    model: models.MLModel = zntrack.zn.deps()

    logspace: bool = zntrack.zn.params(True)
    stop: float = zntrack.zn.params(3.0)
    factor: float = zntrack.zn.params(0.001)
    num: int = zntrack.zn.params(100)

    seed: int = zntrack.zn.params(1234)
    energies: pd.DataFrame = zntrack.zn.plots(
        # x="x",
        # y="y",
        # x_label="stdev of randomly displaced atoms",
        # y_label="predicted energy",
    )

    def post_init(self):
        self.data = utils.helpers.get_deps_if_node(self.data, "atoms")

    def run(self):
        if self.logspace:
            stdev_space = (
                np.logspace(start=0.0, stop=self.stop, num=self.num) * self.factor
            )
        else:
            stdev_space = (
                np.linspace(start=0.0, stop=self.stop, num=self.num) * self.factor
            )

        atoms = self.get_data()
        reference = atoms.copy()
        atoms.calc = self.model.calc

        energies = []

        self.atoms = []

        for stdev in tqdm.tqdm(stdev_space, ncols=70):
            atoms.positions = reference.positions
            atoms.rattle(stdev=stdev, seed=self.seed)
            energies.append(atoms.get_potential_energy())
            self.atoms.append(atoms.copy())

        self.energies = pd.DataFrame({"y": energies, "x": stdev_space})


class BoxScaleAnalysis(base.ProcessSingleAtom):
    """Scale all particles and predict energies.

    Attributes
    ----------
    model: The MLModel node that implements the 'predict' method
    atoms: list[Atoms] to predict properties for
    start: int, default = None
        The initial box scale, default value is the original box size.
    stop: float, default = 1.0
        The stop value for the generated space of stdev points
    num: int, default = 100
        The size of the generated space of stdev points
    """

    model: models.MLModel = zntrack.zn.deps()
    mapping: base.Mapping = zntrack.zn.nodes(None)

    stop: float = zntrack.zn.params(2.0)
    num: int = zntrack.zn.params(100)
    start: float = zntrack.zn.params(1)

    energies: pd.DataFrame = zntrack.zn.plots(
        # x="x",
        # y="y",
        # x_label="Scale factor of the initial cell",
        # y_label="predicted energy",
    )

    def _post_init_(self):
        self.data = utils.helpers.get_deps_if_node(self.data, "atoms")

    def run(self):
        scale_space = np.linspace(start=self.start, stop=self.stop, num=self.num)

        original_atoms = self.get_data()
        cell = original_atoms.copy().cell
        original_atoms.calc = self.model.calc

        energies = []
        self.atoms = []
        if self.mapping is None:
            scaling_atoms = original_atoms
        else:
            scaling_atoms, molecules = self.mapping.forward_mapping(original_atoms)

        for scale in tqdm.tqdm(scale_space, ncols=70):
            scaling_atoms.set_cell(cell=cell * scale, scale_atoms=True)

            if self.mapping is None:
                eval_atoms = scaling_atoms
            else:
                eval_atoms = self.mapping.backward_mapping(scaling_atoms, molecules)
                # New atoms object, does not have the calculator.
                eval_atoms.calc = self.model.calc

            energies.append(eval_atoms.get_potential_energy())
            self.atoms.append(eval_atoms.copy())

        self.energies = pd.DataFrame({"y": energies, "x": scale_space})


class BoxHeatUp(base.ProcessSingleAtom):
    """Attributes
    ----------
    start_temperature: float
        the temperature to start the analysis.
    stop_temperature: float
        the upper bound of the temperature
    steps: int
        Number of steps between lower and upper temperature
    time_step: float, default = 0.5 fs
        time step of the simulation
    friction: float, default = 0.01
        langevin friction

    """

    start_temperature: float = zntrack.zn.params()
    stop_temperature: float = zntrack.zn.params()
    steps: int = zntrack.zn.params()
    time_step: float = zntrack.zn.params(0.5)
    friction = zntrack.zn.params()
    repeat = zntrack.zn.params((1, 1, 1))

    max_temperature: float = zntrack.zn.params(None)

    flux_data = zntrack.zn.plots()

    model = zntrack.zn.deps()

    plots = zntrack.dvc.outs(zntrack.nwd / "temperature.png")

    def get_atoms(self) -> ase.Atoms:
        atoms: ase.Atoms = self.get_data()
        return atoms.repeat(self.repeat)

    def plot_temperature(self):
        fig, ax = plt.subplots()
        ax.plot(self.flux_data["set_temp"], self.flux_data["meassured_temp"])
        ax.plot(
            self.flux_data["set_temp"],
            self.flux_data["set_temp"],
            color="grey",
            linestyle="--",
        )
        ax.set_ylim(self.start_temperature, self.stop_temperature)
        ax.set_xlabel(r"Target temperature $t_\mathrm{target}$ / K")
        ax.set_ylabel(r"Measured temperature $t_\mathrm{exp}$ / K")
        fig.savefig(self.plots)

    def run(self):
        if self.max_temperature is None:
            self.max_temperature = self.stop_temperature * 1.5
        atoms = self.get_atoms()
        atoms.set_calculator(self.model.calc)
        # Initialize velocities
        MaxwellBoltzmannDistribution(atoms, temperature_K=self.start_temperature)
        # initialize thermostat
        thermostat = Langevin(
            atoms,
            timestep=self.time_step * units.fs,
            temperature_K=self.start_temperature,
            friction=self.friction,
        )
        # Run simulation

        temperature, total_energy = utils.ase_sim.get_energy(atoms)

        energy = []
        self.atoms = []

        with tqdm.trange(
            self.steps,
            desc=utils.ase_sim.get_desc(temperature, total_energy),
            leave=True,
            ncols=120,
        ) as pbar:
            for temp in np.linspace(
                self.start_temperature, self.stop_temperature, self.steps
            ):
                thermostat.run(1)
                thermostat.set_temperature(temperature_K=temp)
                temperature, total_energy = utils.ase_sim.get_energy(atoms)
                pbar.set_description(utils.ase_sim.get_desc(temperature, total_energy))
                pbar.update()
                energy.append([temperature, total_energy, temp])
                if temperature > self.max_temperature:
                    log.critical(
                        "Temperature of the simulation exceeded"
                        f" {self.max_temperature} K. Simulation was stopped."
                    )
                    break
                self.atoms.append(atoms.copy())

        self.flux_data = pd.DataFrame(
            energy, columns=["meassured_temp", "energy", "set_temp"]
        )
        self.flux_data[self.flux_data > self.max_temperature] = self.max_temperature
        self.flux_data.index.name = "step"
        if temperature > self.max_temperature:
            self.steps_before_explosion = len(energy)
        else:
            self.steps_before_explosion = -1

        self.plot_temperature()


def compute_trans_forces(mol):
    """Compute translational forces of a molecule."""

    all_forces = np.sum(mol.get_forces(), axis=0)
    masses = mol.get_masses()
    mol_mas = np.sum(masses)
    res = (masses / mol_mas)[:, None] * all_forces
    return res


def compute_intertia_tensor(centered_positions, masses):
    r_sq = np.linalg.norm(centered_positions, ord=2, axis=1) ** 2 * masses
    r_sq = np.sum(r_sq)
    A = np.diag(np.full((3,), r_sq))
    mr_k = centered_positions * masses[:, None]
    B = np.einsum("ki, kj -> ij", centered_positions, mr_k)

    I_ab = A - B
    return I_ab


def compute_rot_forces(mol):
    mol_positions = mol.get_positions()
    mol_positions -= mol.get_center_of_mass()
    masses = mol.get_masses()

    f_x_r = np.sum(np.cross(mol.get_forces(), mol_positions), axis=0)
    I_ab = compute_intertia_tensor(mol_positions, masses)
    I_ab_inv = np.linalg.inv(I_ab)

    mi_ri = masses[:, None] * mol_positions
    res = np.cross(mi_ri, (I_ab_inv @ f_x_r))

    return res


def force_decomposition(atom, mapping):
    # TODO we should only need to do the mapping once
    _, molecules = mapping.forward_mapping(atom)
    full_forces = np.zeros_like(atom.positions)
    atom_trans_forces = np.zeros_like(atom.positions)
    atom_rot_forces = np.zeros_like(atom.positions)
    total_n_atoms = 0

    for molecule in molecules:
        n_atoms = len(molecule)
        full_forces[total_n_atoms : total_n_atoms + n_atoms] = molecule.get_forces()
        atom_trans_forces[total_n_atoms : total_n_atoms + n_atoms] = compute_trans_forces(
            molecule
        )
        atom_rot_forces[total_n_atoms : total_n_atoms + n_atoms] = compute_rot_forces(
            molecule
        )
        total_n_atoms += n_atoms

    atom_vib_forces = full_forces - atom_trans_forces - atom_rot_forces

    return atom_trans_forces, atom_rot_forces, atom_vib_forces


class InterIntraForces(base.AnalyseProcessAtoms):
    """Node for decomposing forces in a system of molecular units into
    translational, rotational and vibrational components.

    The implementation follows the method described in
    https://doi.org/10.26434/chemrxiv-2022-l4tb9
    """

    trans_forces: dict = zntrack.zn.metrics()
    rot_forces: dict = zntrack.zn.metrics()
    vib_forces: dict = zntrack.zn.metrics()

    rot_force_plt = zntrack.dvc.outs(zntrack.nwd / "rot_force.png")
    trans_force_plt = zntrack.dvc.outs(zntrack.nwd / "trans_force.png")
    vib_force_plt = zntrack.dvc.outs(zntrack.nwd / "vib_force.png")

    def get_plots(self):
        fig = get_figure(
            np.linalg.norm(self.true_forces["trans"], axis=-1),
            np.linalg.norm(self.pred_forces["trans"], axis=-1),
            datalabel="",
            xlabel=r"$ab~initio$ forces / eV$ \cdot \AA^{-1}",
            ylabel=r"predicted forces / eV$ \cdot \AA^{-1}",
        )
        fig.savefig(self.trans_force_plt)

        fig = get_figure(
            np.linalg.norm(self.true_forces["rot"], axis=-1),
            np.linalg.norm(self.pred_forces["rot"], axis=-1),
            datalabel="",
            xlabel=r"$ab~initio$ forces / eV$ \cdot \AA^{-1}",
            ylabel=r"predicted forces / eV$ \cdot \AA^{-1}",
        )
        fig.savefig(self.rot_force_plt)

        fig = get_figure(
            np.linalg.norm(self.true_forces["vib"], axis=-1),
            np.linalg.norm(self.pred_forces["vib"], axis=-1),
            datalabel="",
            xlabel=r"$ab~initio$ forces / eV$ \cdot \AA^{-1}",
            ylabel=r"predicted forces / eV$ \cdot \AA^{-1}",
        )
        fig.savefig(self.vib_force_plt)

    def get_metrics(self):
        """Update the metrics."""

        self.trans_forces = {
            "rmse": utils.metrics.root_mean_squared_error(
                np.array(self.true_forces["trans"]), np.array(self.pred_forces["trans"])
            ),
            "mae": utils.metrics.mean_absolute_error(
                np.array(self.true_forces["trans"]), np.array(self.pred_forces["trans"])
            ),
            "max": utils.metrics.maximum_error(
                np.array(self.true_forces["trans"]), np.array(self.pred_forces["trans"])
            ),
        }
        self.rot_forces = {
            "rmse": utils.metrics.root_mean_squared_error(
                np.array(self.true_forces["rot"]), np.array(self.pred_forces["rot"])
            ),
            "mae": utils.metrics.mean_absolute_error(
                np.array(self.true_forces["rot"]), np.array(self.pred_forces["rot"])
            ),
            "max": utils.metrics.maximum_error(
                np.array(self.true_forces["rot"]), np.array(self.pred_forces["rot"])
            ),
        }
        self.vib_forces = {
            "rmse": utils.metrics.root_mean_squared_error(
                np.array(self.true_forces["vib"]), np.array(self.pred_forces["vib"])
            ),
            "mae": utils.metrics.mean_absolute_error(
                np.array(self.true_forces["vib"]), np.array(self.pred_forces["vib"])
            ),
            "max": utils.metrics.maximum_error(
                np.array(self.true_forces["vib"]), np.array(self.pred_forces["vib"])
            ),
        }

    def run(self):
        true_atoms, pred_atoms = self.get_data()
        mapping = BarycenterMapping(data=None)

        true_trans_forces = []
        true_rot_forces = []
        true_vib_forces = []

        for atom in tqdm.tqdm(true_atoms):
            atom_trans_forces, atom_rot_forces, atom_vib_forces = force_decomposition(
                atom, mapping
            )
            true_trans_forces.append(atom_trans_forces)
            true_rot_forces.append(atom_rot_forces)
            true_vib_forces.append(atom_vib_forces)

        true_trans_forces = np.concatenate(true_trans_forces)
        true_rot_forces = np.concatenate(true_rot_forces)
        true_vib_forces = np.concatenate(true_vib_forces)

        pred_trans_forces = []
        pred_rot_forces = []
        pred_vib_forces = []

        for atom in tqdm.tqdm(pred_atoms):
            atom_trans_forces, atom_rot_forces, atom_vib_forces = force_decomposition(
                atom, mapping
            )
            pred_trans_forces.append(atom_trans_forces)
            pred_rot_forces.append(atom_rot_forces)
            pred_vib_forces.append(atom_vib_forces)

        pred_trans_forces = np.concatenate(pred_trans_forces)
        pred_rot_forces = np.concatenate(pred_rot_forces)
        pred_vib_forces = np.concatenate(pred_vib_forces)

        self.pred_forces = {
            "trans": pred_trans_forces,
            "rot": pred_rot_forces,
            "vib": pred_vib_forces,
        }
        self.true_forces = {
            "trans": true_trans_forces,
            "rot": true_rot_forces,
            "vib": true_vib_forces,
        }
        self.get_metrics()
        self.get_plots()


class NaNCheck(base.CheckBase):
    """Check Node to see whether positions, energies or forces become NaN
    during a simulation.
    """

    def check(self, atoms: ase.Atoms) -> bool:
        positions = atoms.positions
        epot = atoms.get_potential_energy()
        forces = atoms.get_forces()

        positions_is_none = np.any(positions is None)
        epot_is_none = epot is None
        forces_is_none = np.any(forces is None)

        unstable = any([positions_is_none, epot_is_none, forces_is_none])
        return unstable


class ConnectivityCheck(base.CheckBase):
    """Check to see whether the covalent connectivity of the system
    changes during a simulation.
    The connectivity is based on ASE's natural cutoffs.

    """

    def _post_init_(self) -> None:
        self.nl = None
        self.first_cm = None

    def initialize(self, atoms):
        self.nl = build_neighbor_list(atoms, self_interaction=False)
        self.first_cm = self.nl.get_connectivity_matrix(sparse=False)
        self.is_initialized = True

    def check(self, atoms: ase.Atoms) -> bool:
        self.nl.update(atoms)
        cm = self.nl.get_connectivity_matrix(sparse=False)

        connectivity_change = np.sum(np.abs(self.first_cm - cm))

        unstable = connectivity_change > 0
        return unstable


class EnergySpikeCheck(base.CheckBase):
    """Check to see whether the potential energy of the system has fallen
    below a minimum or above a maximum threshold.

    Attributes
    ----------
    min_factor: Simulation stops if `E(current) > E(initial) * min_factor`
    max_factor: Simulation stops if `E(current) < E(initial) * max_factor`
    """

    min_factor: float = zntrack.zn.params(0.5)
    max_factor: float = zntrack.zn.params(2.0)

    def _post_init_(self) -> None:
        self.max_energy = None
        self.min_energy = None

    def initialize(self, atoms: ase.Atoms) -> None:
        epot = atoms.get_potential_energy()
        self.max_energy = epot * self.max_factor
        self.min_energy = epot * self.min_factor

    def check(self, atoms: ase.Atoms) -> bool:
        epot = atoms.get_potential_energy()
        # energy is negative, hence sign convention
        if epot < self.max_energy or epot > self.min_energy:
            unstable = True
        else:
            unstable = False
        return unstable


def run_stability_nve(
    atoms: ase.Atoms,
    time_step: float,
    max_steps: int,
    init_temperature: float,
    checks: list[base.CheckBase],
    save_last_n: int,
    rng: typing.Optional[np.random.Generator] = None,
) -> typing.Tuple[int, list[ase.Atoms]]:
    """
    Runs an NVE trajectory for a single configuration until either max_steps
    is reached or one of the checks fails.
    """
    pbar_update = 50
    stable_steps = 0

    MaxwellBoltzmannDistribution(atoms, temperature_K=init_temperature, rng=rng)
    etot, ekin, epot = get_energy_terms(atoms)
    last_n_atoms = deque(maxlen=save_last_n)
    last_n_atoms.append(atoms.copy())

    for check in checks:
        check.initialize(atoms)

    def get_desc():
        """TQDM description."""
        return f"Etot: {etot:.3f} eV  \t Ekin: {ekin:.3f} eV \t Epot {epot:.3f} eV"

    dyn = VelocityVerlet(atoms, timestep=time_step * units.fs)
    with trange(
        max_steps,
        desc=get_desc(),
        leave=True,
        ncols=120,
        position=1,
    ) as pbar:
        for idx in range(max_steps):
            dyn.run(1)
            last_n_atoms.append(atoms.copy())
            etot, ekin, epot = get_energy_terms(atoms)

            if idx % pbar_update == 0:
                pbar.set_description(get_desc())
                pbar.update(pbar_update)

            check_results = [check.check(atoms) for check in checks]
            unstable = any(check_results)
            if unstable:
                stable_steps = idx
                break

    return stable_steps, list(last_n_atoms)


class MDStabilityAnalysis(base.ProcessAtoms):
    """Perform NVE molecular dynamics for all supplied atoms using a trained model.
    Several stability checks can be supplied to judge whether a particular
    trajectory is stable.
    If the check fails, the trajectory is terminated.
    After all trajectories are done, a histogram of the duration of stability is created.

    Attributes
    ----------
    model: A node which implements the `calc` property. Typically an MLModel instance.
    data: list[Atoms] to run MD for for
    max_steps: Maximum number of steps for each trajectory
    time_step: MD integration time step
    initial_temperature: Initial velocities are drawn from a maxwell boltzman
    distribution.
    save_last_n: how many configurations before the instability should be saved
    bins: number of bins in the histogram
    seed: seed for the MaxwellBoltzmann distribution
    """

    model = zntrack.zn.deps()
    max_steps: int = zntrack.zn.params()
    checks: list[zntrack.Node] = zntrack.zn.nodes()
    time_step: float = zntrack.zn.params(0.5)
    initial_temperature: float = zntrack.zn.params(300)
    save_last_n: int = zntrack.zn.params(1)
    bins: int = zntrack.zn.params(None)
    seed: int = zntrack.zn.params(0)

    traj_file: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "trajectory.h5")
    plots_dir: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "plots")
    stable_steps_df: pd.DataFrame = zntrack.zn.plots()

    @property
    def atoms(self) -> typing.List[ase.Atoms]:
        return znh5md.ASEH5MD(self.traj_file).get_atoms_list()

    def get_plots(self, stable_steps: int) -> None:
        """Create figures for all available data."""
        if self.bins is None:
            self.bins = int(np.ceil(len(stable_steps) / 100))
        counts, bin_edges = np.histogram(stable_steps, self.bins)

        self.plots_dir.mkdir()

        label_hist = get_histogram_figure(
            bin_edges,
            counts,
            datalabel="NVE",
            xlabel="Number of stable time steps",
            ylabel="Occurences",
        )
        label_hist.savefig(self.plots_dir / "hist.png")

    def run(self) -> None:
        data_lst = self.get_data()
        calculator = self.model.calc
        rng = default_rng(self.seed)

        stable_steps = []

        db = znh5md.io.DataWriter(self.traj_file)
        db.initialize_database_groups()
        unstable_atoms = []

        for ii in tqdm.trange(
            0, len(data_lst), desc="Atoms", leave=True, ncols=120, position=0
        ):
            atoms = data_lst[ii].copy()
            atoms.calc = calculator
            n_steps, last_n_atoms = run_stability_nve(
                atoms,
                self.time_step,
                self.max_steps,
                self.initial_temperature,
                checks=self.checks,
                save_last_n=self.save_last_n,
                rng=rng,
            )
            unstable_atoms.extend(last_n_atoms)
            stable_steps.append(n_steps)
        db.add(
            znh5md.io.AtomsReader(
                unstable_atoms,
                frames_per_chunk=self.save_last_n,
                step=1,
            )
        )

        self.get_plots(stable_steps)
        self.stable_steps_df = pd.DataFrame({"stable_steps": np.array(stable_steps)})
