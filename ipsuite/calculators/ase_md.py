import logging
import pathlib
import typing

import ase
import ase.constraints
import ase.geometry
import numpy as np
import pandas as pd
import znh5md
import zntrack
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from tqdm import trange

from ipsuite import base

log = logging.getLogger(__name__)


class CheckBase(zntrack.Node):
    def initialize(self, atoms):
        pass

    def check(self, atoms):
        raise NotImplementedError

    def get_metric(self):
        return None

    def get_desc(self):
        return None


class TemperatureCheck(CheckBase):
    """Calculate and check teperature during a MD simulation

    Attributes
    ----------
    max_temperature: float
        maximum temperature, when reaching it simulation will be stopped
    """

    max_temperature = zntrack.zn.params(10000.0)

    def check(self, atoms):
        ekin = atoms.get_kinetic_energy() / len(atoms)
        self.temperature = ekin / (1.5 * units.kB)
        unstable = self.temperature > self.max_temperature
        return unstable

    def get_metric(self):
        return {"temperature": self.temperature}

    def get_desc(self):
        return f"Temp: {self.temperature:.3f} K"


class LagevinThermostat(zntrack.Node):
    """Initialize the lagevin thermostat

    Attributes
    ----------
    time_step: float
        time step of simulation

    temperature: float
        temperature in K to simulate at

    friction: float
        friction of the Langevin simulator

    """

    time_step = zntrack.zn.params()
    temperature = zntrack.zn.params()
    friction = zntrack.zn.params()

    def get_thermostat(self, atoms):
        self.time_step *= units.fs
        thermostat = Langevin(
            atoms=atoms,
            timestep=self.time_step,
            temperature_K=self.temperature,
            friction=self.friction,
        )
        return thermostat


def get_energy(atoms: ase.Atoms) -> float:
    """Compute the total energy.

    Parameters
    ----------
    atoms: ase.Atoms
        Atoms objects for which energy will be calculated

    Returns
    -------
    np.squeeze(total): float
        total energy of the system

    """
    e_tot = atoms.get_total_energy() / len(atoms)

    return np.squeeze(e_tot)


class ASEMD(base.ProcessSingleAtom):
    """Class to run a MD simulation with ASE.

    Attributes
    ----------
    atoms_lst: list
        list of atoms objects to start simulation from
    start_id: int
        starting id to pick from list of atoms
    ase_calculator: ase.calculator
        ase calculator to use for simulation
    checker_list: list[CheckNodes]
        checker, which tracks various metrics and stops the
        simulation after a threshold is exceeded.
    thermostat_node: ase dynamics
        dynamics method used for simulation
    init_temperature: float
        temperature in K to initialize velocities
    init_velocity: np.array()
        starting velocities to continue a simulation
    steps: int
        number of steps to simulate
    sampling_rate: int
        number of sample runs
    metrics_dict:
        saved total energy and all metrics from the check nodes
    repeat: float
        number of repeats
    traj_file: Path
        path where to save the trajectory
    dump_rate: int, default=1000
        Keep a cache of the last 'dump_rate' atoms and
        write them to the trajectory file every 'dump_rate' steps.
    """

    ase_calculator = zntrack.zn.deps()
    checker_list = zntrack.zn.nodes()
    thermostat_node = zntrack.zn.nodes()

    steps = zntrack.zn.params()
    init_temperature = zntrack.zn.params(None)
    init_velocity = zntrack.zn.params(None)
    sampling_rate = zntrack.zn.params(1)
    repeat = zntrack.zn.params((1, 1, 1))
    dump_rate = zntrack.zn.params(1000)

    metrics_dict = zntrack.zn.plots()

    steps_before_stopping: int = zntrack.zn.metrics()
    velocity_cach = zntrack.zn.metrics()

    traj_file: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "trajectory.h5")

    def get_constraint(self):
        return []

    def get_atoms(self) -> ase.Atoms:
        atoms: ase.Atoms = self.get_data()
        return atoms.repeat(self.repeat)

    @property
    def atoms(self) -> typing.List[ase.Atoms]:
        return znh5md.ASEH5MD(self.traj_file).get_atoms_list()

    def run(self):
        """Run the simulation."""
        atoms = self.get_atoms()
        atoms.calc = self.ase_calculator.calculator

        if self.init_temperature is not None and self.init_velocity is None:
            # Initialize velocities
            MaxwellBoltzmannDistribution(atoms, temperature_K=self.init_temperature)
        elif self.init_velocity is not None and self.init_temperature is None:
            # Continue with last md step
            atoms.set_velocities(self.init_velocity)
        else:
            raise ValueError("init_temperature or init_velocity has to be specified.")

        # initialize thermostat
        time_step = self.thermostat_node.time_step
        thermostat = self.thermostat_node.get_thermostat(atoms=atoms)

        # initialize Atoms calculator and metrics_dict
        _ = get_energy(atoms)
        metrics_dict = {"energy": []}
        for checker in self.checker_list:
            _ = checker.check(atoms)
            metric = checker.get_metric()
            if metric is not None:
                for key in metric.keys():
                    metrics_dict[key] = []

        # Run simulation
        total_fs = int(self.steps * time_step * self.sampling_rate)

        atoms.set_constraint(self.get_constraint())

        atoms_cache = []

        db = znh5md.io.DataWriter(self.traj_file)
        db.initialize_database_groups()

        with trange(
            total_fs,
            leave=True,
            ncols=120,
        ) as pbar:
            for idx in range(self.steps):
                desc = []
                stop = []
                thermostat.run(self.sampling_rate)
                metrics_dict["energy"].append(get_energy(atoms))

                for checker in self.checker_list:
                    stop.append(checker.check(atoms))
                    if stop[-1]:
                        log.critical(
                            f"\n {type(checker).__name__} returned false."
                            "Simulation was stopped."
                        )
                    metric = checker.get_metric()
                    if metric is not None:
                        for key, val in metric.items():
                            metrics_dict[key].append(val)
                        desc.append(checker.get_desc())

                if any(stop):
                    self.steps_before_stopping = len(metrics_dict["energy"])
                    break
                else:
                    atoms_cache.append(atoms.copy())
                    if len(atoms_cache) == self.dump_rate:
                        db.add(
                            znh5md.io.AtomsReader(
                                atoms_cache,
                                frames_per_chunk=self.dump_rate,
                                step=1,
                                time=self.sampling_rate,
                            )
                        )
                        atoms_cache = []

                    energy = metrics_dict["energy"][-1]
                    desc.append(f"E: {energy:.3f} eV")
                    if idx % (1 / time_step) == 0:
                        pbar.set_description("\t".join(desc))
                        pbar.update(self.sampling_rate)

        # save the last configurations
        db.add(
            znh5md.io.AtomsReader(
                atoms_cache,
                frames_per_chunk=self.dump_rate,
                step=1,
                time=self.sampling_rate,
            )
        )

        self.velocity_cach = atoms.get_velocities()
        self.metrics_dict = pd.DataFrame(metrics_dict)

        self.metrics_dict.index.name = "step"
        self.steps_before_stopping = -1


class FixedSphereASEMD(ASEMD):
    """Attributes
    ----------
    atom_id: int
        The id to use as the center of the sphere to fix.
        If None, the closed atom to the center will be picked.
    radius: float
    """

    atom_id = zntrack.zn.params(None)
    selected_atom_id = zntrack.zn.outs()
    radius = zntrack.zn.params()

    def get_constraint(self):
        atoms = self.get_atoms()
        r_ij, d_ij = ase.geometry.get_distances(atoms.get_positions())
        if self.atom_id is not None:
            self.selected_atom_id = self.atom_id
        else:
            _, dist = ase.geometry.get_distances(
                atoms.get_positions(), np.diag(atoms.get_cell() / 2)
            )
            self.selected_atom_id = np.argmin(dist)

        if isinstance(self.selected_atom_id, np.generic):
            self.selected_atom_id = self.selected_atom_id.item()

        indices = np.nonzero(d_ij[self.selected_atom_id] < self.radius)[0]
        return ase.constraints.FixAtoms(indices=indices)
