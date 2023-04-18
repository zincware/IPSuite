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
