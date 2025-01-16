import ase
import networkx as nx
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator

from ipsuite.geometry import graphs
from ipsuite.geometry.graphs import edges_from_atoms


def closest_atom_to_center(atoms):
    closest_atom = np.argmin(
        np.linalg.norm(atoms.positions - np.diag(atoms.cell) / 2, ord=2, axis=1)
    )
    return closest_atom


def sort_atomic_edges(edges, idx):
    idxs = np.nonzero(np.any(edges == idx, axis=1))[0]
    current_edges = edges[idxs]

    unsorted = current_edges[:, 1] == idx
    unsorted = np.nonzero(unsorted)[0]
    for idx in unsorted:
        current_edges[idx][[0, 1]] = current_edges[idx][[1, 0]]
    return current_edges


def displace_neighbors(mol, edges):
    dist = mol.get_distance(edges[0], edges[1], vector=True)
    pdist = mol.get_distance(edges[0], edges[1], True, vector=True)

    displacement = dist - pdist
    mol.positions[edges[1]] -= displacement


def unwrap(atoms, edges, idx):
    G = graphs.atoms_to_graph(atoms)
    edges = nx.traversal.bfs_edges(G, idx)
    for e in edges:
        displace_neighbors(atoms, e)


def unwrap_system(atoms: ase.Atoms, components: list[np.ndarray], forces=None) -> list[ase.Atom]:
    """Molecules in a system which extend across periodic boundaries are mapped such that
    they are connected but dangle out of the cell.
    Mapping to the side where the fragment of molecule is closest
    to the cell center is preferred.
    Can be reversed by joining the returned molecules
    and calling the `atoms.wrap()` method.
    """
    molecules = []
    for component in components:
        mol = atoms[component].copy()
        if forces is not None:
            results = {"forces": forces[component]}
            mol.calc = SinglePointCalculator(mol, **results)
            
        elif atoms.calc is not None:
            results = {"forces": atoms.get_forces()[component]}
            if "forces_uncertainty" in atoms.calc.results.keys():
                f_unc = atoms.calc.results["forces_uncertainty"][component]
                results["forces_uncertainty"] = f_unc

            if "forces_ensemble" in atoms.calc.results.keys():
                f_ens = atoms.calc.results["forces_ensemble"][component]
                results["forces_ensemble"] = f_ens

            mol.calc = SinglePointCalculator(mol, **results)
        edges = edges_from_atoms(mol)
        closest_atom = closest_atom_to_center(mol)
        unwrap(mol, edges, idx=closest_atom)
        molecules.append(mol)

    return molecules
