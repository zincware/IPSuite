import dataclasses
import logging

import ase
import ase.constraints
import ase.geometry
import numpy as np

log = logging.getLogger(__name__)


@dataclasses.dataclass
class FixedSphereConstraint:
    """Attributes
    ----------
    atom_id: int
        The id to use as the center of the sphere to fix.
        If None, the closed atom to the center will be picked.
    atom_type: str, optional
        The type of the atom to fix. E.g. if
        atom_type = H, atom_id = 1, the first
        hydrogen atom will be fixed. If None,
        the first atom will be fixed, no matter the type.
    radius: float
    """

    radius: float
    atom_id: int | None = None
    atom_type: str | None = None

    def __post_init__(self):
        if self.atom_type is not None and self.atom_id is None:
            raise ValueError("If atom_type is given, atom_id must be given as well.")

    def get_selected_atom_id(self, atoms: ase.Atoms) -> int:
        if self.atom_type is not None:
            return np.where(np.array(atoms.get_chemical_symbols()) == self.atom_type)[0][
                self.atom_id
            ]

        elif self.atom_id is not None:
            return self.atom_id
        else:
            _, dist = ase.geometry.get_distances(
                atoms.get_positions(), np.diag(atoms.get_cell() / 2)
            )
            return np.argmin(dist)

    def get_constraint(self, atoms):
        r_ij, d_ij = ase.geometry.get_distances(
            atoms.get_positions(), cell=atoms.cell, pbc=True
        )
        selected_atom_id = self.get_selected_atom_id(atoms)

        indices = np.nonzero(d_ij[selected_atom_id] < self.radius)[0]
        return ase.constraints.FixAtoms(indices=indices)


@dataclasses.dataclass
class FixedLayerConstraint:
    """Class to fix a layer of atoms within a MD
        simulation

    Attributes
    ----------
    upper_limit: float
        all atoms with a lower z pos will be fixed.
    lower_limit: float
        all atoms with a higher z pos will be fixed.
    """

    upper_limit: float
    lower_limit: float

    def get_constraint(self, atoms):
        z_coordinates = atoms.positions[:, 2]

        self.indices = np.where(
            (self.lower_limit <= z_coordinates) & (z_coordinates <= self.upper_limit)
        )[0]

        return ase.constraints.FixAtoms(indices=self.indices)


@dataclasses.dataclass
class FixedBondLengthConstraint:
    """Fix the Bondlength between two atoms

    Attributes
    ----------
    atom_id_1: int
        index of atom 1
    atom_id_2: int
        index of atom 2

    Returns
    -------
    ase.constraints.FixBondLengths
        Constraint that fixes the bond Length between atom_id_1 and atom_id_2
    """

    atom_id_1: int
    atom_id_2: int

    def get_constraint(self, atoms: ase.Atoms):
        return ase.constraints.FixBondLength(self.atom_id_1, self.atom_id_2)


@dataclasses.dataclass
class HookeanConstraint:
    """Applies a Hookean (spring) force between pairs of atoms.

    Attributes
    ----------
    atom_ids: list[tuple]
        List of atom indices that need to be constrained.
        Example: Fix only atoms in water with bonds if the IDs are
        as follows:  H=0, H=1, O=2
        Then the following atom_ids list is needed: [(0, 2), (1, 2)]
    k: float
        Hookes law (spring) constant to apply when distance exceeds threshold_length.
        Units of eV A^-2.
        # TODO: Allow different k for each pair
    rt: float
        The threshold length below which there is no force.
        # TODO: Allow different rt for each pair


    Returns
    -------
    list[ase.constraints.Hookean]
        List of constraints that fixes the bond Length between the
        molecules in the atom_id tuples.
    """

    atom_ids: list[tuple]
    k: float
    rt: float

    def get_pairs(self, molecule: tuple):
        atoms = len(molecule)
        pairs = [(i, j) for i in range(0, atoms) for j in range(i + 1, atoms)]
        return pairs

    def get_constraint(self, atoms: ase.Atoms):
        constraints = []
        for molecule in self.atom_ids:
            pairs = self.get_pairs(molecule)
            for pair in pairs:
                constraints.append(
                    ase.constraints.Hookean(
                        int(molecule[pair[0]]), int(molecule[pair[1]]), self.k, self.rt
                    )
                )

        return constraints
