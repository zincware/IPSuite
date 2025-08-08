"""Constraint classes that use atom selections."""

import dataclasses

import ase
from ase.constraints import FixAtoms, FixBondLength, FixBondLengths

from ipsuite.interfaces import AtomConstraint, AtomSelector


@dataclasses.dataclass
class FixAtomsConstraint(AtomConstraint):
    """Fix atoms in a selection during optimization or dynamics.

    Parameters
    ----------
    selection : AtomSelector
        Atom selection strategy that determines which atoms to fix.
    """

    selection: AtomSelector

    def get_constraint(self, atoms: ase.Atoms) -> FixAtoms:
        """Get the FixAtoms constraint for the selected atoms.

        Parameters
        ----------
        atoms : ase.Atoms
            Atomic structure to apply constraint to.

        Returns
        -------
        FixAtoms
            ASE constraint that fixes the selected atoms.
        """
        indices = self.selection.select(atoms)
        if not indices:
            raise ValueError("No atoms selected for FixAtoms constraint")
        return FixAtoms(indices=indices)


@dataclasses.dataclass
class FixBondLengthConstraint(AtomConstraint):
    """Fix bond length between two selected atoms.

    Parameters
    ----------
    atom1_selection : AtomSelector
        Selection for the first atom. Must select exactly one atom.
    atom2_selection : AtomSelector
        Selection for the second atom. Must select exactly one atom.
    """

    atom1_selection: AtomSelector
    atom2_selection: AtomSelector

    def get_constraint(self, atoms: ase.Atoms) -> FixBondLength:
        """Get the FixBondLength constraint for the selected atom pair.

        Parameters
        ----------
        atoms : ase.Atoms
            Atomic structure to apply constraint to.

        Returns
        -------
        FixBondLength
            ASE constraint that fixes the bond length between selected atoms.

        Raises
        ------
        ValueError
            If either selection doesn't select exactly one atom.
        """
        indices1 = self.atom1_selection.select(atoms)
        indices2 = self.atom2_selection.select(atoms)

        if len(indices1) != 1:
            raise ValueError(
                f"atom1_selection must select exactly 1 atom, got {len(indices1)}"
            )
        if len(indices2) != 1:
            raise ValueError(
                f"atom2_selection must select exactly 1 atom, got {len(indices2)}"
            )

        return FixBondLength(indices1[0], indices2[0])


@dataclasses.dataclass
class FixBondLengthsConstraint(AtomConstraint):
    """Fix multiple bond lengths using atom selections.

    Parameters
    ----------
    bond_pairs : list[tuple[AtomSelector, AtomSelector]]
        List of tuples, each containing two AtomSelector objects that should
        each select exactly one atom to form a bond pair.
    """

    bond_pairs: list[tuple[AtomSelector, AtomSelector]]

    def get_constraint(self, atoms: ase.Atoms) -> FixBondLengths:
        """Get the FixBondLengths constraint for multiple selected atom pairs.

        Parameters
        ----------
        atoms : ase.Atoms
            Atomic structure to apply constraint to.

        Returns
        -------
        FixBondLengths
            ASE constraint that fixes bond lengths between selected atom pairs.

        Raises
        ------
        ValueError
            If any selection doesn't select exactly one atom.
        """
        pairs = []

        for i, (sel1, sel2) in enumerate(self.bond_pairs):
            indices1 = sel1.select(atoms)
            indices2 = sel2.select(atoms)

            if len(indices1) != 1:
                raise ValueError(
                    f"Bond pair {i}, atom1 selection must select exactly 1 atom, got {len(indices1)}"
                )
            if len(indices2) != 1:
                raise ValueError(
                    f"Bond pair {i}, atom2 selection must select exactly 1 atom, got {len(indices2)}"
                )

            pairs.append([indices1[0], indices2[0]])

        if not pairs:
            raise ValueError("No bond pairs provided for FixBondLengths constraint")

        return FixBondLengths(pairs)
