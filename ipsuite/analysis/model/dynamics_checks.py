import collections
import dataclasses
import typing

import ase
import numpy as np
from ase.geometry import conditional_find_mic
from ase.neighborlist import build_neighbor_list, natural_cutoffs

from ipsuite import base
from ipsuite.utils.ase_sim import get_energy
from scipy import sparse


@dataclasses.dataclass
class DebugCheck(base.Check):
    """A check that interrupts the dynamics after a fixed amount of iterations.
    For testing purposes.

    Attributes
    ----------
    n_iterations: int
        number of iterations before stopping
    """

    n_iterations: int = 10

    def __post_init__(self) -> None:
        self.counter = 0
        self.status = self.__class__.__name__

    def check(self, atoms):
        if self.counter >= self.n_iterations:
            return True
        self.counter += 1
        return False


@dataclasses.dataclass
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


@dataclasses.dataclass
class ConnectivityCheck(base.Check):
    """Check to see whether the covalent connectivity of the system
    changes during a simulation.
    The connectivity is based on ASE's natural cutoffs.
    The pair of atoms which triggered this check will be converted to
    Lithium for easy visibility

    """

    bonded_min_dist: float = 0.6
    bonded_max_dist: float = 2.0

    def __post_init__(self) -> None:
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


@dataclasses.dataclass
class EnergySpikeCheck(base.Check):
    """Check to see whether the potential energy of the system has fallen
    below a minimum or above a maximum threshold.

    Attributes
    ----------
    min_factor: Simulation stops if `E(current) > E(initial) * min_factor`
    max_factor: Simulation stops if `E(current) < E(initial) * max_factor`
    """

    min_factor: float = 0.5
    max_factor: float = 2.0

    max_energy: float | None = None
    min_energy: float | None = None

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


@dataclasses.dataclass
class TemperatureCheck(base.Check):
    """Calculate and check teperature during a MD simulation

    Attributes
    ----------
    max_temperature: float
        maximum temperature, when reaching it simulation will be stopped
    """

    max_temperature: float = 10000.0

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


@dataclasses.dataclass
class ThresholdCheck(base.Check):
    """Calculate and check a given threshold and std during a MD simulation

    Compute the standard deviation of the selected property.
    If the property is off by more than a selected amount from the
    mean, the simulation will be stopped.
    Furthermore, the simulation will be stopped if the property
    exceeds a threshold value.

    Attributes
    ----------
    key: str
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

    key: str = "energy_uncertainty"
    max_std: float = None
    window_size: int = 500
    max_value: float = None
    minimum_window_size: int = 1
    larger_only: bool = False

    def __post_init__(self):
        if self.max_std is None and self.max_value is None:
            raise ValueError("Either max_std or max_value must be set")
        self.values = collections.deque(maxlen=self.window_size)

    def get_value(self, atoms):
        """Get the value of the property to check.
        Extracted into method so it can be subclassed.
        """
        return np.max(atoms.calc.results[self.key])

    def get_quantity(self):
        if self.max_value is None:
            return f"{self.key}-threshold-std-{self.max_std}"
        else:
            return f"{self.key}-threshold-max-{self.max_value}"

    def check(self, atoms) -> bool:
        value = atoms.calc.results[self.key]
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
                f"StandardDeviationCheck for {self.key} triggered by"
                f" '{np.max(self.values[-1]):.3f}' > max_value {self.max_value}"
            )
            return True

        elif self.max_std is not None and np.max(distance) > self.max_std * std:
            self.status = (
                f"StandardDeviationCheck for '{self.key}' triggered by"
                f" '{np.max(self.values[-1]):.3f}' for '{mean:.3f} +-"
                f" {std:.3f}' and max value '{self.max_value}'"
            )
            return True
        else:
            self.status = (
                f"StandardDeviationCheck for '{self.key}' passed with"
                f" '{np.max(self.values[-1]):.3f}' for '{mean:.3f} +-"
                f" {std:.3f}' and max value '{self.max_value}'"
            )
            return False


@dataclasses.dataclass
class ReflectionCheck(base.Check):
    """
    A class to check and handle the reflection of atoms in a simulation.

    Parameters
    ----------
    cutoff_plane : float
        The z-coordinate of the cutoff plane. If None, `cutoff_plane_dist` must be specified.
    additive_idx : int
        Index of the additive atom to monitor. If None, all atoms are considered for penetration check.
    cutoff_plane_dist : float
        Distance from the maximum z-coordinate of atoms to define the cutoff plane. Used if `cutoff_plane` is None.
    cutoff_plane_skin : float
        Skin distance added to the cutoff plane for determining reflection criteria.
        
    Attributes:
    ----------
    reflected : bool
        Indicates if atoms have been reflected.
    cutoff_penetrated : bool
        Indicates if the cutoff plane has been penetrated by atoms.
    z_max : float
        Maximum z-coordinate of atoms in the initial configuration.
    """
    cutoff_plane_height: float = None
    additive_idx: typing.List[int] = None
    cutoff_plane_dist: float = None
    cutoff_plane_skin: float = 1.5
    del_reflected_atoms: bool = False
    
    def initialize(self, atoms: ase.Atoms) -> None:
        self.reflected = False
        self.cutoff_penetrated = False
        
        z_pos = atoms.positions[:,2]
        if self.additive_idx is None:
            z_max = np.max(z_pos)
        else:
            z_max = np.max(np.delete(z_pos, self.additive_idx))
            
        if self.cutoff_plane_height is None and self.cutoff_plane_dist is None:
            raise ValueError("Either cutoff_plane or cutoff_plane_dist has to be specified.")
        elif self.cutoff_plane_dist is not None:
            if self.cutoff_plane_height is not None:
                raise ValueError("Specify either cutoff_plane or cutoff_plane_dist, not both.")
            self.cutoff_plane = z_max + self.cutoff_plane_dist
        else:
            self.cutoff_plane = self.cutoff_plane_height
        
    def check(self, atoms) -> bool:
        z_pos = atoms.positions[:,2]
        idxs = np.where(z_pos > self.cutoff_plane)[0]
        
        if self.additive_idx is None:
            self.cutoff_penetrated = True
        else:
            additive_z_pos = z_pos[self.additive_idx]
            if not self.cutoff_penetrated and additive_z_pos < self.cutoff_plane:
                self.cutoff_penetrated = True
            
        if self.cutoff_penetrated and len(idxs) != 0:
            self.reflected = True
            
        if self.reflected:
            nl = build_neighbor_list(atoms, self_interaction=False)
            matrix = nl.get_connectivity_matrix()
            n_components, component_list = sparse.csgraph.connected_components(matrix)

            self.del_atom_idxs = []
            del_mol_idxs = []
            for atom_idx in idxs:
                mol_idx = component_list[atom_idx]
                if mol_idx not in del_mol_idxs:
                    del_mol_idxs.append(mol_idx)
                    self.del_atom_idxs.extend([i for i in range(len(component_list)) if component_list[i] == mol_idx])
                    
            self.out_velo = atoms.get_velocities()[self.del_atom_idxs]
            self.status = (
                    f"Molecule/s {del_mol_idxs} with Atom(s) {self.del_atom_idxs} was/were reflected and deleted.\n"
                    f"Atoms idx = {self.del_atom_idxs}: v = {self.out_velo}"
                )

            return True

        return False
    
    def mod_atoms(self, atoms):
        if self.reflected and self.del_reflected_atoms:
            del atoms[self.del_atom_idxs]
            return True
        else:
            return False
    
    
    def get_value(self, atoms):
        """Get the value of the property to check.
        Extracted into method so it can be subclassed.
        """
        return self.out_velo if self.reflected else None

    def get_quantity(self):
            return f"out_velo"
        
        


@dataclasses.dataclass
class PlanePenetrationCheck(base.Check):
    """

    """
    cutoff_plane_height: float = None
    additive_idx: typing.List[int] = None
    cutoff_plane_dist: float = None
    cutoff_plane_skin: float = 1.5
    del_reflected_atoms: bool = False

    def initialize(self, atoms: ase.Atoms) -> None:
        self.cutoff_penetrated = False
        self.reflected = False
        
        z_pos = atoms.positions[:,2]
        z_max = np.max(np.delete(z_pos, self.additive_idx))
            
        if self.cutoff_plane_height is None and self.cutoff_plane_dist is None:
            raise ValueError("Either cutoff_plane or cutoff_plane_dist has to be specified.")
        elif self.cutoff_plane_dist is not None:
            if self.cutoff_plane_height is not None:
                raise ValueError("Specify either cutoff_plane or cutoff_plane_dist, not both.")
            self.cutoff_plane = z_max + self.cutoff_plane_dist
        else:
            self.cutoff_plane = self.cutoff_plane_height
        
    def check(self, atoms) -> bool:
        z_pos = atoms.positions[:,2]
        idxs = np.where(z_pos > self.cutoff_plane)[0]

        additive_z_pos = z_pos[self.additive_idx]
        if additive_z_pos < self.cutoff_plane:
            self.cutoff_penetrated = True
            
        if self.cutoff_penetrated and len(idxs) != 0:
            self.reflected = True
            
        if self.reflected:
            nl = build_neighbor_list(atoms, self_interaction=False)
            matrix = nl.get_connectivity_matrix()
            n_components, component_list = sparse.csgraph.connected_components(matrix)

            self.del_atom_idxs = []
            del_mol_idxs = []
            for atom_idx in idxs:
                mol_idx = component_list[atom_idx]
                if mol_idx not in del_mol_idxs:
                    del_mol_idxs.append(mol_idx)
                    self.del_atom_idxs.extend([i for i in range(len(component_list)) if component_list[i] == mol_idx])
                    
            self.out_velo = atoms.get_velocities()[self.del_atom_idxs]
            self.status = (
                    f"Molecule/s {del_mol_idxs} with Atom(s) {self.del_atom_idxs} was/were reflected and deleted.\n"
                    f"Atoms idx = {self.del_atom_idxs}: v = {self.out_velo}"
                )

            return True
        
        if self.cutoff_penetrated:
            self.status = (
                    f"Atom penetrated cutoff plane."
                )
            return True
        
        return False

    def mod_atoms(self, atoms):
        if self.reflected and self.del_reflected_atoms:
            del atoms[self.del_atom_idxs]
            return True
        else:
            return False
        
    def get_value(self, atoms):
        """Get the value of the property to check.
        Extracted into method so it can be subclassed.
        """
        return self.out_velo if self.reflected else None

    def get_quantity(self):
            return f"out_velo"