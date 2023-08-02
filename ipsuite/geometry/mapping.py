"""Molecule Mapping using networkx"""
import typing

import ase

from ipsuite import base
from ipsuite.geometry import barycenter_coarse_grain, graphs, unwrap


class BarycenterMapping(base.Mapping):
    """Node that "coarse grains" each molecule in a configuration into its center of mass.
    Useful for operations affecting intermolecular distance,
    but not intramolecular distances.
    """

    _components = None

    def forward_mapping(
        self, atoms: ase.Atoms
    ) -> typing.Tuple[ase.Atoms, list[ase.Atoms]]:
        if self._components is None:
            components = graphs.identify_molecules(atoms)
        else:
            components = self._components

        if self.frozen:
            self._components = components
        molecules = unwrap.unwrap_system(atoms, components)
        cg_atoms = barycenter_coarse_grain.coarse_grain_to_barycenter(molecules)
        return cg_atoms, molecules

    def backward_mapping(
        self, cg_atoms: ase.Atoms, molecules: list[ase.Atoms]
    ) -> list[ase.Atoms]:
        atoms = barycenter_coarse_grain.barycenter_backmapping(cg_atoms, molecules)
        return atoms
