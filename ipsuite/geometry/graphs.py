import ase
import networkx as nx
import numpy as np
from ase.neighborlist import build_neighbor_list, natural_cutoffs


def atoms_to_graph(atoms: ase.Atoms, cutoffs: dict[str, float] | None = None) -> nx.Graph:
    """Converts ASE Atoms into a Graph based on their
    bond connectivity.

    Args:
        atoms (ase.Atoms): Atoms instance to convert
        cutoffs (dict[str, float] | None): cutoffs of each atom.
            Dictionary with keys for the symbols and values of the cutoff radii.
            If None, use the `ase.data.covalent_radii`. Default: None

    Returns:
        G (nx.Graph): Connectivity graph
    """
    # This can be optimized by reusing the NL!
    if cutoffs is not None:
        cutoffs = natural_cutoffs(atoms, **cutoffs)
    nl = build_neighbor_list(atoms, self_interaction=False, cutoffs=cutoffs)
    cm = nl.get_connectivity_matrix(sparse=False)
    G = nx.from_numpy_array(cm)
    return G


def identify_molecules(
    atoms: ase.Atoms, cutoffs: dict[str, float] | None = None
) -> list[np.ndarray]:
    """Identifies molecules in a structure based on the connected subgraphs.

    Args:
        atoms (ase.Atoms): Atoms instance to identify molecules in
        cutoffs (dict[str, float] | None): cutoffs of each element.
            Dictionary with keys for the symbols and values of the cutoff radii.
            If None, use the `ase.data.covalent_radii`. Default: None

    Returns:
        c_list (np.ndarray): Array of lists of connected atom indices
    """
    G = atoms_to_graph(atoms, cutoffs=cutoffs)
    components = nx.connected_components(G)
    c_list = [np.array(list(c)) for c in components]
    return c_list


def split_molecule(a0, a1, atoms):
    G = atoms_to_graph(atoms)

    try:
        G.remove_edge(a0, a1)
    except ValueError:
        print(f"Atom {a0} and {a1} are not bonded. Pick bonded atoms.")

    components = nx.connected_components(G)
    c_list = [np.array(list(c)) for c in components]
    return c_list


def edges_from_atoms(atoms: ase.Atoms) -> np.ndarray:
    """Returns the graph edges of a molecular graph."""
    G = atoms_to_graph(atoms)
    edges = np.array(G.edges)
    return edges
