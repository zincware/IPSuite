"""Molecule Mapping using networkx"""

from ipsuite import base
from ipsuite.geometry import barycenter_coarse_grain, graphs, unwrap


class BarycenterMapping(base.Mapping):
    def forward_mapping(self, atoms):
        components = graphs.identify_molecules(atoms)
        molecules = unwrap.unwrap_system(atoms, components)
        cg_atoms = barycenter_coarse_grain.coarse_grain_to_barycenter(molecules)
        return cg_atoms, molecules

    def backward_mapping(self, cg_atoms, molecules):
        atoms = barycenter_coarse_grain.barycenter_backmapping(cg_atoms, molecules)
        return atoms
