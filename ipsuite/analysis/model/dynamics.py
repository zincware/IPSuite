import functools
import logging
import pathlib
import typing
from collections import deque

import ase
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import znh5md
import zntrack
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from numpy.random import default_rng
from tqdm import trange

from ipsuite import base, models, utils
from ipsuite.analysis.ensemble import plot_with_uncertainty
from ipsuite.analysis.model.plots import get_histogram_figure
from ipsuite.utils.ase_sim import freeze_copy_atoms
from ipsuite.utils.md import get_energy_terms

log = logging.getLogger(__name__)


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

    model: models.MLModel = zntrack.deps()
    model_outs = zntrack.outs_path(zntrack.nwd / "model/")

    logspace: bool = zntrack.params(True)
    stop: float = zntrack.params(3.0)
    factor: float = zntrack.params(0.001)
    num: int = zntrack.params(100)

    seed: int = zntrack.params(1234)
    energies: pd.DataFrame = zntrack.plots(
        # x="x",
        # y="y",
        # x_label="stdev of randomly displaced atoms",
        # y_label="predicted energy",
    )

    def run(self):
        self.model_outs.mkdir(parents=True, exist_ok=True)
        (self.model_outs / "outs.txt").write_text("Lorem Ipsum")

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
        atoms.calc = self.model.get_calculator(directory=self.model_outs)

        energies = []

        self.atoms = []

        for stdev in tqdm.tqdm(stdev_space, ncols=70):
            atoms.positions = reference.positions
            atoms.rattle(stdev=stdev, seed=self.seed)
            energies.append(atoms.get_potential_energy())
            self.atoms.append(freeze_copy_atoms(atoms))

        self.energies = pd.DataFrame({"y": energies, "x": stdev_space})


class BoxScale(base.ProcessSingleAtom):
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

    model: models.MLModel = zntrack.deps()
    model_outs = zntrack.outs_path(zntrack.nwd / "model")
    mapping: base.Mapping = zntrack.deps(None)

    stop: float = zntrack.params(2.0)
    num: int = zntrack.params(100)
    start: float = zntrack.params(1)

    plot = zntrack.outs_path(zntrack.nwd / "energy.png")

    energies: pd.DataFrame = zntrack.plots(
        x="x",
        y="y",
        x_label="Scale factor of the initial cell",
        y_label="predicted energy",
    )

    def run(self):
        self.model_outs.mkdir(parents=True, exist_ok=True)
        (self.model_outs / "outs.txt").write_text("Lorem Ipsum")
        scale_space = np.linspace(start=self.start, stop=self.stop, num=self.num)

        original_atoms = self.get_data()
        cell = original_atoms.copy().cell
        original_atoms.calc = self.model.get_calculator(directory=self.model_outs)

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
                eval_atoms.calc = original_atoms.calc

            energies.append(eval_atoms.get_potential_energy())
            self.atoms.append(freeze_copy_atoms(eval_atoms))

        self.energies = pd.DataFrame({"y": energies, "x": scale_space})

        if "energy_uncertainty" in self.atoms[0].calc.results:
            fig, ax, _ = plot_with_uncertainty(
                {
                    "std": np.std(
                        [a.calc.results["energy_uncertainty"] for a in self.atoms]
                    ),
                    "mean": self.energies["y"],
                },
                x=self.energies["x"],
                ylabel="predicted energy",
                xlabel="Scale factor of the initial cell",
            )
        else:
            fig, ax = plt.subplots()
            ax.plot(self.energies["x"], self.energies["y"])
            ax.set_xlabel("Scale factor of the initial cell")
            ax.set_ylabel("predicted energy")
        fig.savefig(self.plot, bbox_inches="tight")


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

    start_temperature: float = zntrack.params()
    stop_temperature: float = zntrack.params()
    steps: int = zntrack.params()
    time_step: float = zntrack.params(0.5)
    friction = zntrack.params()
    repeat = zntrack.params((1, 1, 1))

    max_temperature: float = zntrack.params(None)

    flux_data = zntrack.plots()

    model = zntrack.deps()
    model_outs = zntrack.outs_path(zntrack.nwd / "model")

    plots = zntrack.outs_path(zntrack.nwd / "temperature.png")

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
        self.model_outs.mkdir(parents=True, exist_ok=True)
        (self.model_outs / "outs.txt").write_text("Lorem Ipsum")
        if self.max_temperature is None:
            self.max_temperature = self.stop_temperature * 1.5
        atoms = self.get_atoms()
        atoms.calc = self.model.get_calculator(directory=self.model_outs)
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
                self.atoms.append(freeze_copy_atoms(atoms))

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


def run_stability_nve(
    atoms: ase.Atoms,
    time_step: float,
    max_steps: int,
    init_temperature: float,
    checks: list[base.Check],
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
    last_n_atoms.append(freeze_copy_atoms(atoms))

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
            last_n_atoms.append(freeze_copy_atoms(atoms))
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


class MDStability(base.ProcessAtoms):
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

    model = zntrack.deps()
    model_outs = zntrack.outs_path(zntrack.nwd / "model_outs")
    max_steps: int = zntrack.params()
    checks: list[zntrack.Node] = zntrack.deps()
    time_step: float = zntrack.params(0.5)
    initial_temperature: float = zntrack.params(300)
    save_last_n: int = zntrack.params(1)
    bins: int = zntrack.params(None)
    seed: int = zntrack.params(0)

    traj_file: pathlib.Path = zntrack.outs_path(zntrack.nwd / "structures.h5")
    plots_dir: pathlib.Path = zntrack.outs_path(zntrack.nwd / "plots")
    stable_steps_df: pd.DataFrame = zntrack.plots()

    @property
    def atoms(self) -> typing.List[ase.Atoms]:
        with self.state.fs.open(self.traj_file, "rb") as f:
            with h5py.File(f) as file:
                return znh5md.IO(file_handle=file)[:]

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
            ylabel="Occurrences",
        )
        label_hist.savefig(self.plots_dir / "hist.png")

    def run(self) -> None:
        self.model_outs.mkdir(parents=True, exist_ok=True)
        (self.model_outs / "outs.txt").write_text("Lorem Ipsum")
        data_lst = self.get_data()
        calculator = self.model.get_calculator(directory=self.model_outs)
        rng = default_rng(self.seed)

        stable_steps = []

        db = znh5md.IO(self.traj_file)
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
        db.extend(unstable_atoms)

        self.get_plots(stable_steps)
        self.stable_steps_df = pd.DataFrame({"stable_steps": np.array(stable_steps)})
