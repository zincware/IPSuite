import ase
import networkx as nx
import numpy as np
from ase.neighborlist import build_neighbor_list


def atoms_to_graph(atoms: ase.Atoms) -> nx.Graph:
    """Converts ASE Atoms into a Graph based on their
    bond connectivity.
    """
    # This can be optimized by reusing the NL!
    nl = build_neighbor_list(atoms, self_interaction=False)
    cm = nl.get_connectivity_matrix(sparse=False)
    G = nx.from_numpy_array(cm)
    return G


def identify_molecules(atoms: ase.Atoms) -> list[np.ndarray]:
    """Identifies molecules in a structure based on the connected subgraphs."""
    G = atoms_to_graph(atoms)
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
