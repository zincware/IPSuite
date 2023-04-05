import logging
import pathlib
import typing
from collections import deque

import ase
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
from ase.neighborlist import build_neighbor_list
from numpy.random import default_rng
from tqdm import trange

from ipsuite import base, models, utils
from ipsuite.analysis.bin_property import get_histogram_figure
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
