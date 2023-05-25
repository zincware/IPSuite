import collections

import ase
import numpy as np
import zntrack
from ase.neighborlist import build_neighbor_list

from ipsuite import base
from ipsuite.utils.ase_sim import get_energy
from ase.geometry import conditional_find_mic

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


def setdiff2d(arr1, arr2):
    idx = (arr1[:, None] != arr2).any(-1).all(1)
    return arr1[idx]

def check_distances(a, idx_i, idx_j, d_min=None, d_max=None):
    p1 = a.positions[idx_i]
    p2 = a.positions[idx_j]
    _ , dists = conditional_find_mic(p1-p2, a.cell, a.pbc)
    unstable = False
    if d_min:
        unstable = unstable or np.min(dists) < d_min
    if d_max:
        unstable = unstable or np.max(dists) > d_max
    return unstable



class ConnectivityCheck(base.CheckBase):
    """Check to see whether the covalent connectivity of the system
    changes during a simulation.
    The connectivity is based on ASE's natural cutoffs.

    """
    bonded_min_dist: float = zntrack.zn.params(0.6)
    bonded_max_dist: float = zntrack.zn.params(2.0)
    nonbonded_H_min_dist: float = zntrack.zn.params(1.1)
    nonbonded_other_min_dist: float = zntrack.zn.params(1.6)

    def _post_init_(self) -> None:
        self.nl = None
        self.first_cm = None

    def initialize(self, atoms):
        from ase.neighborlist import natural_cutoffs, NeighborList
        cutoffs = natural_cutoffs(atoms, mult=1.5)
        nl = build_neighbor_list(atoms, cutoffs=cutoffs, skin=0.0, self_interaction=False, bothways=False)
        first_cm = nl.get_connectivity_matrix(sparse=True)
        self.indices = np.vstack(first_cm.nonzero()).T
        self.idx_i, self.idx_j = self.indices.T

        cutoffs = natural_cutoffs(atoms, mult=0.7)
        self.contact_nl = build_neighbor_list(atoms, cutoffs=cutoffs, skin=0.5, self_interaction=False, bothways=False)
        self.is_initialized = True

    def check(self, atoms: ase.Atoms) -> bool:
        unstable = False
        bonded_check = check_distances(atoms, self.idx_i, self.idx_j, d_min=self.bonded_min_dist, d_max=self.bonded_max_dist)
        unstable = unstable or bonded_check

        self.contact_nl.update(atoms)
        nearest_cm = self.contact_nl.get_connectivity_matrix(sparse=True)
        nearest_indices = np.vstack(nearest_cm.nonzero()).T
        diff_indices = setdiff2d(nearest_indices, self.indices)

        if len(diff_indices) > 0:
            diff_idx_i, diff_idx_j = diff_indices.T
            Zij = np.stack([atoms.numbers[diff_idx_i], atoms.numbers[diff_idx_j]]).T

            both_H = np.any((Zij - 1) == 0, axis=1)
            if np.any(both_H):
                H_idx_i, H_idx_j = diff_indices[both_H].T
                H_check = check_distances(atoms, H_idx_i, H_idx_j, d_min=self.nonbonded_H_min_dist, d_max=None)
                unstable = unstable or H_check

            not_both_H = np.ones_like(both_H).astype(bool)
            not_both_H[both_H] = False
            if np.any(not_both_H):
                other_idx_i, other_idx_j = diff_indices[not_both_H].T
                not_both_H_check = check_distances(atoms, other_idx_i, other_idx_j, d_min=self.nonbonded_other_min_dist, d_max=None)
                unstable = unstable or not_both_H_check

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

        if unstable:
            self.status = (
                f"Temperature Check failed: last {self.temperature} >"
                f" {self.max_temperature}"
            )
        else:
            self.status = f"Temperature Check {self.temperature} < {self.max_temperature}"

        return unstable

    def get_metric(self):
        return {"temperature": self.temperature}

    def __str__(self):
        return self.status

    def get_desc(self) -> str:
        return str(self)


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
        Minimum number of steps to average over before checking the standard deviation
    larger_only: bool, optional
        Only check the standard deviation of points that are larger than the mean.
        E.g. useful for uncertainties, where a lower uncertainty is not a problem.
    """

    value: str = zntrack.zn.params()
    max_std: float = zntrack.zn.params(None)
    window_size: int = zntrack.zn.params(500)
    max_value: float = zntrack.zn.params(None)
    minimum_window_size: int = zntrack.zn.params(50)
    larger_only: bool = zntrack.zn.params(False)

    def _post_init_(self):
        if self.max_std is None and self.max_value is None:
            raise ValueError("Either max_std or max_value must be set")

    def _post_load_(self) -> None:
        self.values = collections.deque(maxlen=self.window_size)
        self.status = self.__class__.__name__

    def get_value(self, atoms):
        """Get the value of the property to check.

        Extracted into method so it can be subclassed.
        """
        return atoms.calc.results[self.value]

    def check(self, atoms) -> bool:
        value = self.get_value(atoms)
        self.values.append(value)
        mean = np.mean(self.values)
        std = np.std(self.values)

        distance = value - mean
        if self.larger_only:
            distance = np.abs(distance)

        if self.max_value is not None and value > self.max_value:
            # max value trigger is independent of the window size.
            return True

        if len(self.values) < self.minimum_window_size:
            return False

        if self.max_std is not None and distance > self.max_std * std:
            self.status = (
                f"StandardDeviationCheck for '{self.value}' triggered by"
                f" '{self.values[-1]:.3f}' for '{np.mean(self.values):.3f} +-"
                f" {np.std(self.values):.3f}' and max value '{self.max_value}'"
            )
            return True
        return False

    def __str__(self) -> str:
        return self.status

    def get_desc(self) -> str:
        return str(self)
