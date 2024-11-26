"""Molecule Mapping using networkx"""

import dataclasses
import typing as t

import ase

from ipsuite.geometry import barycenter_coarse_grain, graphs, unwrap


@dataclasses.dataclass
class BarycenterMapping:
    """Node that "coarse grains" each molecule in a configuration into its center of mass.
    Useful for operations affecting intermolecular distance,
    but not intramolecular distances.


    `Mapping` nodes can be used in a more functional manner when initialized
    with `data=None` outside the project graph.
    In that case, one can use the mapping methods but the Node itself does not store
    the transformed configurations.

    Attributes
    ----------
    frozen: bool
        If True, the neighbor list is only constructed for the first configuration.
        The indices of the molecules will be frozen for all configurations.
    """

    frozen: bool = False

    _components: t.Any | None = None

    def forward_mapping(self, atoms: ase.Atoms) -> tuple[ase.Atoms, list[ase.Atoms]]:
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
