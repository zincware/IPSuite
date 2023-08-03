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

    def initialize(self, atoms: ase.Atoms) -> None:
        self.is_initialized = True

    def check(self, atoms: ase.Atoms) -> bool:
        positions = atoms.positions
        epot = atoms.get_potential_energy()
        forces = atoms.get_forces()

        positions_is_none = np.any(positions is None)
        epot_is_none = epot is None
        forces_is_none = np.any(forces is None)

        if any([positions_is_none, epot_is_none, forces_is_none]):
            self.status = (
                "NaN check failed: last iterationpositions energy or forces = NaN"
            )
            return True
        else:
            self.status = "No NaN occurred"
            return False


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
        if connectivity_change > 0:
            self.status = (
                "Connectivity check failed: last iteration"
                "covalent connectivity of the system changed"
            )
            return True
        else:
            self.status = "covalent connectivity of the system is intact"
            return False


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
        if epot < self.max_energy:
            self.status = (
                "Energy spike check failed: last iteration"
                f"E {epot} > E_max {self.max_energy}"
            )
            return True

        elif epot > self.min_energy:
            self.status = (
                "Energy spike check failed: last iteration"
                f"E {epot} < E_min {self.min_energy}"
            )
            return True
        else:
            self.status = "No energy spike occurred"
            return False


class TemperatureCheck(base.CheckBase):
    """Calculate and check teperature during a MD simulation

    Attributes
    ----------
    max_temperature: float
        maximum temperature, when reaching it simulation will be stopped
    """

    max_temperature: float = zntrack.zn.params(10000.0)

    def initialize(self, atoms: ase.Atoms) -> None:
        self.is_initialized = True

    def check(self, atoms):
        self.temperature, _ = get_energy(atoms)

        if self.temperature > self.max_temperature:
            self.status = (
                "Temperature Check failed last iteration"
                f"T {self.temperature} K > T_max {self.max_temperature} K"
            )
            return True
        else:
            self.status = (
                f"Temperature Check: T {self.temperature} K <"
                f"T_max {self.max_temperature} K"
            )
            return False


class ThresholdCheck(base.CheckBase):
    """Calculate and check a given threshold and std during a MD simulation

    Compute the standard deviation of the selected property.
    If the property is off by more than a selected amount from the
    mean, the simulation will be stopped.
    Furthermore, the simulation will be stopped if the property
    exceeds a threshold value.

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
    minimum_window_size: int, optional
        Minimum number of steps to average over before checking the standard deviation.
        Also minimum number of steps to run, before the simulation can be stopped.
    larger_only: bool, optional
        Only check the standard deviation of points that are larger than the mean.
        E.g. useful for uncertainties, where a lower uncertainty is not a problem.
    """

    value: str = zntrack.zn.params()
    max_std: float = zntrack.zn.params(None)
    window_size: int = zntrack.zn.params(500)
    max_value: float = zntrack.zn.params(None)
    minimum_window_size: int = zntrack.zn.params(1)
    larger_only: bool = zntrack.zn.params(False)

    def _post_init_(self):
        if self.max_std is None and self.max_value is None:
            raise ValueError("Either max_std or max_value must be set")

    def _post_load_(self) -> None:
        self.values = collections.deque(maxlen=self.window_size)

    def get_value(self, atoms):
        """Get the value of the property to check.
        Extracted into method so it can be subclassed.
        """
        return np.max(atoms.calc.results[self.value])

    def get_quantity(self):
        if self.max_value is None:
            return f"{self.value}-threshold-std-{self.max_std}"
        else:
            return f"{self.value}-threshold-max-{self.max_value}"

    def check(self, atoms) -> bool:
        # self.status = "None"
        value = atoms.calc.results[self.value]
        self.values.append(value)
        mean = np.mean(self.values)
        std = np.std(self.values)

        distance = value - mean
        if self.larger_only:
            distance = np.abs(distance)

        if len(self.values) < self.minimum_window_size:
            return False

        if self.max_value is not None and np.max(value) > self.max_value:
            self.status = (
                f"StandardDeviationCheck for {self.value} triggered by"
                f" '{np.max(self.values[-1]):.3f}' > max_value {self.max_value}"
            )
            return True

        elif self.max_std is not None and distance > self.max_std * std:
            self.status = (
                f"StandardDeviationCheck for '{self.value}' triggered by"
                # dependent on self.value self.values can be an ndarray
                # then this status gives an error message np.max(self.values)
                # should be the best option here. but I'm not sure if it
                # is what you intended.
                f" '{np.max(self.values[-1]):.3f}' for '{mean:.3f} +-"
                f" {std:.3f}' and max value '{self.max_value}'"
            )
            return True
        else:
            self.status = (
                f"StandardDeviationCheck for '{self.value}' passed with"
                f" '{np.max(self.values[-1]):.3f}' for '{mean:.3f} +-"
                f" {std:.3f}' and max value '{self.max_value}'"
            )
            return False
