import logging
import pathlib
import typing

import ase
import ase.constraints
import ase.geometry
import numpy as np
import pandas as pd
import zntrack
from ase import units
from ase.io import write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from tqdm import trange

from ipsuite import base

log = logging.getLogger(__name__)


def print_energy(atoms: ase.Atoms) -> typing.Tuple[float, float]:
    """Compute the temperature and the total energy.

    Parameters
    ----------
    atoms: ase.Atoms
        Atoms objects for which energy will be calculated

    Returns
    -------
    temperature: float
        temperature of the system
    np.squeeze(total): float
        total energy of the system

    """
    epot = atoms.get_potential_energy() / len(atoms)
    ekin = atoms.get_kinetic_energy() / len(atoms)

    temperature = ekin / (1.5 * units.kB)
    total = epot + ekin

    return temperature, np.squeeze(total)


class ASEMD(base.ProcessSingleAtom):
    """Class to run a MD simulation with ASE.

    Attributes
    ----------
    atoms_lst: list
        list of atoms objects to start simulation from
    start_id: int
        starting id to pick from list of atoms
    model: MLModel
        Model to use for simulation
    temperature: float
        temperature in K to simulate at
    time_step: float
        time step of simulation
    friction: float
        friction of the Langevin simulator
    steps: int
        number of steps to simulate
    sampling_rate: int
        number of sample runs
    max_temperature: float
        maximum temperature, when reaching it simulation will be stopped
    flux_data:
        saved temperature and total energy
    repeat: float
        number of repeats
    traj_file: Path
        path where to save the trajectory
    """

    model = zntrack.zn.deps()
    temperature = zntrack.zn.params()
    time_step = zntrack.zn.params()
    friction = zntrack.zn.params()
    steps = zntrack.zn.params()
    sampling_rate = zntrack.zn.params()
    max_temperature = zntrack.zn.params(10000.0)
    flux_data = zntrack.zn.plots()  # temperature / energy
    repeat = zntrack.zn.params((1, 1, 1))
    steps_before_explosion: int = zntrack.zn.metrics()

    traj_file: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "trajectory.extxyz")

    def get_constraint(self):
        return []

    def get_atoms(self) -> ase.Atoms:
        atoms: ase.Atoms = self.get_data()
        return atoms.repeat(self.repeat)

    @property
    def atoms(self) -> typing.List[ase.Atoms]:
        return list(ase.io.iread(self.traj_file))

    def run(self):
        """Run the simulation."""
        atoms = self.get_atoms()
        atoms.set_calculator(self.model.calc)
        # Initialize velocities
        MaxwellBoltzmannDistribution(atoms, temperature_K=self.temperature)
        # initialize thermostat
        thermostat = Langevin(
            atoms,
            timestep=self.time_step * units.fs,
            temperature_K=self.temperature,
            friction=self.friction,
        )
        # Run simulation

        energy = []
        temperature, total_energy = print_energy(atoms)
        total_fs = int(self.steps * self.time_step * self.sampling_rate)

        atoms.set_constraint(self.get_constraint())

        def get_desc():
            """TQDM description."""
            return (
                f"Temp: {temperature:.3f} K \t Energy {total_energy:.3f} eV - (TQDM"
                " in fs)"
            )

        atoms_cache = []

        with trange(
            total_fs,
            desc=get_desc(),
            leave=True,
            ncols=120,
        ) as pbar:
            for idx in range(self.steps):
                thermostat.run(self.sampling_rate)
                temperature, total_energy = print_energy(atoms)
                energy.append([temperature, total_energy])
                atoms_cache.append(atoms.copy())
                write(
                    self.traj_file,
                    atoms_cache,
                    format="extxyz",
                    append=True,
                )
                atoms_cache = []
                if idx % (1 / self.time_step) == 0:
                    pbar.set_description(get_desc())
                    pbar.update(self.sampling_rate)
                if temperature > self.max_temperature:
                    log.critical(
                        "Temperature of the simulation exceeded"
                        f" {self.max_temperature} K. Simulation was stopped."
                    )
                    break
        self.flux_data = pd.DataFrame(energy, columns=["temperature", "energy"])
        self.flux_data.index.name = "step"
        if temperature > self.max_temperature:
            self.steps_before_explosion = len(energy)
        else:
            self.steps_before_explosion = -1


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
