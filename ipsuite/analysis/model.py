import contextlib
import logging
import pathlib

import ase
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
import zntrack
from ase import units
from ase.calculators.calculator import PropertyNotImplementedError
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from scipy.interpolate import interpn

from ipsuite import base, models, utils

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
    start: int, default = None
        The initial box scale, default value is the original box size.
    """

    model: models.MLModel = zntrack.zn.deps()
    mapping: base.Mapping = zntrack.zn.nodes(None)

    stop: float = zntrack.zn.params(2.0)
    num: int = zntrack.zn.params(100)
    start: float = zntrack.zn.params(None)

    energies: pd.DataFrame = zntrack.zn.plots(
        # x="x",
        # y="y",
        # x_label="Scale factor of the initial cell",
        # y_label="predicted energy",
    )

    def post_init(self):
        self.data = utils.helpers.get_deps_if_node(self.data, "atoms")
        if self.start is None:
            self.start = 1.0

    def run(self):
        scale_space = (
            np.linspace(start=self.start, stop=self.stop, num=self.num)
        )

        atoms = self.get_data()
        cell = atoms.copy().cell
        atoms.calc = self.model.calc

        energies = []
        self.atoms = []

        for scale in tqdm.tqdm(scale_space, ncols=70):
            if self.mapping is not None:
                atoms, molecules = self.mapping.forward_mapping(atoms)
            atoms.set_cell(cell=cell * scale, scale_atoms=True)

            if self.mapping is not None:
                atoms = self.mapping.backward_mapping(atoms, molecules)
                # New atoms object, does not have the calculator.
                atoms.calc = self.model.calc
            energies.append(atoms.get_potential_energy())
            self.atoms.append(atoms.copy())

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
