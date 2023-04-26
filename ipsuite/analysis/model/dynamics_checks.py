import collections

import ase
import numpy as np
import zntrack
from ase.neighborlist import build_neighbor_list

from ipsuite import base
from ipsuite.utils.ase_sim import get_energy


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

        return any([positions_is_none, epot_is_none, forces_is_none])


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

        return connectivity_change > 0


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
        return epot < self.max_energy or epot > self.min_energy


class TemperatureCheck(base.CheckBase):
    """Calculate and check teperature during a MD simulation

    Attributes
    ----------
    max_temperature: float
        maximum temperature, when reaching it simulation will be stopped
    """

    max_temperature: float = zntrack.zn.params(10000.0)

    def check(self, atoms):
        self.temperature, _ = get_energy(atoms)
        unstable = self.temperature > self.max_temperature
        return unstable

    def get_metric(self):
        return {"temperature": self.temperature}

    def get_desc(self):
        return f"Temp: {self.temperature:.3f} K"


class StandardDeviationCheck(base.CheckBase):
    """Calculate and check standard deviation during a MD simulation

    Compute the standard deviation of the selected property.
    If the property is off by more than a selected amount from the
    mean, the simulation will be stopped.

    Attributes
    ----------
    value: str
        name of the property to check
    max_std: float, optional
        Maximum number of standard deviations away from the mean to stop the simulation.
        Roughly the value corresponds to the following percentiles:
            {1: 68%, 2: 95%, 3: 99.7%}
    window_size: int, optional
        Number of steps to average over
    max_value: float, optional
        Maximum value of the property to check before the simulation is stopped
    """

    value: str = zntrack.zn.params()
    max_std: float = zntrack.zn.params(None)
    window_size: int = zntrack.zn.params(500)
    max_value: float = zntrack.zn.params(None)

    def _post_init_(self):
        if self.max_std is None and self.max_value is None:
            raise ValueError("Either max_std or max_value must be set")

        self.values = collections.deque(maxlen=self.window_size)

    def check(self, atoms):
        value = atoms.calc.results[self.value]
        self.values.append(value)
        mean = np.mean(self.values)
        std = np.std(self.values)
        if self.max_std is not None and np.abs(value - mean) > self.max_std * std:
            return True
        if self.max_value is not None and value > self.max_value:
            return True

    def __str__(self) -> str:
        return (
            f"StandardDeviationCheck triggered by {self.value[-1]} for"
            f" {np.mean(self.values):.3f} +- {np.std(self.values):.3f} and max value:"
            f" {self.max_value}"
        )

    def get_desc(self) -> str:
        return str(self)
