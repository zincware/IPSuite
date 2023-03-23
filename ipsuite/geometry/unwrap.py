import ase
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator

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
    for edge in edges:
        dist = mol.get_distance(edge[0], edge[1], vector=True)
        pdist = mol.get_distance(edge[0], edge[1], True, vector=True)

        displacement = dist - pdist
        mol.positions[edge[1]] -= displacement


def unwrap(atoms, edges, idx):
    # TODO this should probably be width first, not depth first
    current_edges = sort_atomic_edges(edges, idx)
    displace_neighbors(atoms, current_edges)

    next_idxs = current_edges[:, 1]

    mask = np.all(edges != idx, axis=1)
    filtered_edges = edges[mask]

    for next_idx in next_idxs:
        unwrap(atoms, filtered_edges, next_idx)


def unwrap_system(atoms: ase.Atoms, components: list[np.ndarray]) -> list[ase.Atom]:
    """Molecules in a system which extend across periodic bounaries are mapped such that
    they are connected but dangle out of the cell.
    Mapping to the side where the fragment of molecule is closest
    to the cell center is preferred.
    Can be reversed by joining the returned molecules
    and calling the `atoms.wrap()` method.
    """
    molecules = []
    for component in components:
        mol = atoms[component].copy()
        mol.calc = SinglePointCalculator(mol, forces=atoms.get_forces()[component])
        edges = edges_from_atoms(mol)
        closest_atom = closest_atom_to_center(atoms)
        unwrap(mol, edges, idx=closest_atom)
        molecules.append(mol)

    return molecules
