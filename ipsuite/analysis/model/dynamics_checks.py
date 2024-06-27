import collections

import ase
import numpy as np
import zntrack
from ase.geometry import conditional_find_mic
from ase.neighborlist import build_neighbor_list, natural_cutoffs

from ipsuite import base
from ipsuite.utils.ase_sim import get_energy


class NaNCheck(base.Check):
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


class ConnectivityCheck(base.Check):
    """Check to see whether the covalent connectivity of the system
    changes during a simulation.
    The connectivity is based on ASE's natural cutoffs.

    """
    bonded_min_dist: float = zntrack.params(0.6)
    bonded_max_dist: float = zntrack.params(2.0)

    def _post_init_(self) -> None:
        self.nl = None
        self.first_cm = None

    def initialize(self, atoms):
        cutoffs = natural_cutoffs(atoms, mult=1.0)
        nl = build_neighbor_list(
            atoms, cutoffs=cutoffs, skin=0.0, self_interaction=False, bothways=False
        )
        first_cm = nl.get_connectivity_matrix(sparse=True)
        self.indices = np.vstack(first_cm.nonzero()).T
        self.idx_i, self.idx_j = self.indices.T

        self.is_initialized = True

    def check(self, atoms: ase.Atoms) -> bool:
        p1 = atoms.positions[self.idx_i]
        p2 = atoms.positions[self.idx_j]
        _, dists = conditional_find_mic(p1 - p2, atoms.cell, atoms.pbc)

        unstable = False
        if self.bonded_min_dist:
            min_dist = np.min(dists)
            too_close = min_dist < self.bonded_min_dist
            unstable = unstable or too_close

            if too_close:
                min_idx = np.argmin(dists)
                first_atom = self.idx_i[min_idx]
                second_atom = self.idx_j[min_idx]

                atoms.numbers[first_atom] = 3
                atoms.numbers[second_atom] = 3

        if self.bonded_max_dist:
            max_dist = np.max(dists)
            too_far = max_dist > self.bonded_max_dist
            unstable = unstable or too_far

            if too_far:
                max_idx = np.argmax(dists)
                first_atom = self.idx_i[max_idx]
                second_atom = self.idx_j[max_idx]

                atoms.numbers[first_atom] = 3
                atoms.numbers[second_atom] = 3

        if unstable:
            self.status = (
                "Connectivity check failed: last iteration"
                "covalent connectivity of the system changed"
            )
            return True
        else:
            self.status = "covalent connectivity of the system is intact"
            return False


class EnergySpikeCheck(base.Check):
    """Check to see whether the potential energy of the system has fallen
    below a minimum or above a maximum threshold.

    Attributes
    ----------
    min_factor: Simulation stops if `E(current) > E(initial) * min_factor`
    max_factor: Simulation stops if `E(current) < E(initial) * max_factor`
    """

    min_factor: float = zntrack.params(0.5)
    max_factor: float = zntrack.params(2.0)

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


class TemperatureCheck(base.Check):
    """Calculate and check teperature during a MD simulation

    Attributes
    ----------
    max_temperature: float
        maximum temperature, when reaching it simulation will be stopped
    """

    max_temperature: float = zntrack.params(10000.0)

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


class ThresholdCheck(base.Check):
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

    value: str = zntrack.params()
    max_std: float = zntrack.params(None)
    window_size: int = zntrack.params(500)
    max_value: float = zntrack.params(None)
    minimum_window_size: int = zntrack.params(1)
    larger_only: bool = zntrack.params(False)

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

        elif self.max_std is not None and np.max(distance) > self.max_std * std:
            self.status = (
                f"StandardDeviationCheck for '{self.value}' triggered by"
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
