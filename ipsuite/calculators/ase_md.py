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
from ipsuite.utils.ase_sim import freeze_copy_atoms, get_energy

log = logging.getLogger(__name__)


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

    time_step: int = zntrack.zn.params()
    temperature: float = zntrack.zn.params()
    friction: float = zntrack.zn.params()

    def get_thermostat(self, atoms):
        self.time_step *= units.fs
        thermostat = Langevin(
            atoms=atoms,
            timestep=self.time_step,
            temperature_K=self.temperature,
            friction=self.friction,
        )
        return thermostat


class ASEMD(base.ProcessSingleAtom):
    """Class to run a MD simulation with ASE.

    Attributes
    ----------
    calculator: ase.calculator
        ase calculator to use for simulation
    checker_list: list[CheckNodes]
        checker, which tracks various metrics and stops the
        simulation after a threshold is exceeded.
    thermostat: ase dynamics
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

    calculator = zntrack.zn.deps()
    checker_list: list = zntrack.zn.nodes(None)
    thermostat: LagevinThermostat = zntrack.zn.nodes()

    steps: int = zntrack.zn.params()
    init_temperature: float = zntrack.zn.params(None)
    init_velocity = zntrack.zn.params(None)
    sampling_rate = zntrack.zn.params(1)
    repeat = zntrack.zn.params((1, 1, 1))
    dump_rate = zntrack.zn.params(1000)

    metrics_dict = zntrack.zn.plots()

    steps_before_stopping = zntrack.zn.metrics()
    velocity_cache = zntrack.zn.metrics()

    traj_file: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "trajectory.h5")

    def get_constraint(self):
        return []

    def get_atoms(self) -> ase.Atoms:
        atoms: ase.Atoms = self.get_data()
        return atoms.repeat(self.repeat)

    @property
    def atoms(self) -> typing.List[ase.Atoms]:
        return znh5md.ASEH5MD(self.traj_file).get_atoms_list()

    def run(self):  # noqa: C901
        """Run the simulation."""
        if self.checker_list is None:
            self.checker_list = []
        atoms = self.get_atoms()
        atoms.calc = self.calculator
        if (self.init_velocity is None) and (self.init_temperature is None):
            self.init_temperature = self.thermostat.temperature

        if self.init_temperature is not None:
            # Initialize velocities
            MaxwellBoltzmannDistribution(atoms, temperature_K=self.init_temperature)
        else:
            # Continue with last md step
            atoms.set_velocities(self.init_velocity)

        # initialize thermostat
        time_step = self.thermostat.time_step
        thermostat = self.thermostat.get_thermostat(atoms=atoms)

        # initialize Atoms calculator and metrics_dict
        _, _ = get_energy(atoms)
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
                _, energy = get_energy(atoms)
                metrics_dict["energy"].append(energy)

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

                atoms_cache.append(freeze_copy_atoms(atoms))
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

                if any(stop):
                    self.steps_before_stopping = len(metrics_dict["energy"])
                    break

        db.add(
            znh5md.io.AtomsReader(
                atoms_cache,
                frames_per_chunk=self.dump_rate,
                step=1,
                time=self.sampling_rate,
            )
        )

        self.velocity_cache = atoms.get_velocities()
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
