import networkx as nx
import numpy as np
from ase.neighborlist import build_neighbor_list


def atoms_to_graph(atoms):
    # This can be optimized by reusing the NL!
    nl = build_neighbor_list(atoms, self_interaction=False)
    cm = nl.get_connectivity_matrix(sparse=False)
    G = nx.from_numpy_array(cm)
    return G


def identify_molecules(atoms):
    G = atoms_to_graph(atoms)
    components = nx.connected_components(G)
    c_list = [np.array(list(c)) for c in components]
    return c_list


def edges_from_atoms(atoms):
    G = atoms_to_graph(atoms)
    edges = np.array(G.edges)
    return edges
