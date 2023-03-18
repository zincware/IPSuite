"""Molecule Mapping using smiles and networkx"""


import ase
import networkx as nx
import numpy as np
import tqdm
import zntrack
from ase.constraints import FixBondLengths
from rdkit import Chem
from rdkit.Chem import AllChem

from ipsuite import base


class MoleculeMapping(base.ProcessAtoms):
    smiles: str = zntrack.zn.params()
    n_molecules: int = zntrack.zn.params()
    threshold: float = zntrack.zn.params(2.0)

    graphs = zntrack.zn.outs()

    def get_allowed_bonds(self) -> dict:
        """Get allowed bonds for each atom type


        Returns
        -------
        dict:
            Dictionary of allowed bonds for each atom type.
            An example of the dictionary is given below.
            The first key is the atom type and the second key is a dictionary
            of allowed bonds for that atom type with the maximum number of bonds.
            {
                'C': {'C': 2, 'H': 3, 'N': 2},
                'N': {'C': 3},
                'B': {'F': 4},
                'F': {'B': 1},
                'H': {'C': 1}
            }

        """
        mol = Chem.MolFromSmiles(self.smiles)
        mol = AllChem.AddHs(mol)

        bonds = {}

        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            if atom.GetSymbol() not in bonds:
                bonds[atom.GetSymbol()] = {}
            neighbors = atom.GetNeighbors()
            symbols = [neighbor.GetSymbol() for neighbor in neighbors]
            for element in set(symbols):
                count = bonds[atom.GetSymbol()].get(element, 0)
                if symbols.count(element) > count:
                    bonds[atom.GetSymbol()][element] = symbols.count(element)

        return bonds

    def get_graph(self, atoms, bonds) -> nx.Graph:
        """Get a graph from atoms

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms to generate a graph from
        threshold : float
            Threshold for the distance between atoms to be considered a bond
        count : int
            Number of isolated molecules.
        bonds : dict
            Dictionary of allowed bonds for each atom type.
        """
        sizes = []
        thresholds = []
        threshold = self.threshold
        for _ in range(100):  # 100 trials
            distances = atoms.get_all_distances()
            graph = nx.Graph()
            for idx, atom_i in enumerate(atoms):
                graph.add_node(idx, atom=atom_i)

            np.fill_diagonal(distances, np.inf)
            # TODO do it per symbol and max neighbors
            symbols = np.array(atoms.get_chemical_symbols())
            for symbol in set(symbols):
                # only consider allowed bonds via smiles
                possible_dij = distances.copy()
                possible_dij[symbols != symbol] = np.inf
                possible_dij[:, ~np.isin(symbols, list(bonds[symbol]))] = np.inf
                graph.add_edges_from(
                    [tuple(x) for x in np.argwhere(possible_dij < threshold)]
                )
            size = len(list(nx.connected_components(graph)))
            sizes.append(size)
            thresholds.append(threshold)
            if size == self.n_molecules:
                return graph
            elif size > self.n_molecules:
                threshold += 0.01
            else:
                threshold -= 0.01
        raise ValueError(f"Could not generate a graph {sizes =} with {thresholds =}")

    def run(self) -> None:
        atoms = self.get_data()
        self.graphs = []
        bonds = self.get_allowed_bonds()
        self.graphs.extend(self.get_graph(atoms, bonds) for atoms in tqdm.tqdm(atoms))

    def get_rdkit_molecules(self, item=None):
        if item is None:
            item = slice(None)
        molecules = []
        for graph in self.graphs[item]:
            for mol_ids in nx.connected_components(graph):
                mol_graph = graph.subgraph(mol_ids)
                mapping = {idx: val for val, idx in enumerate(mol_graph.nodes)}
                mol_graph = nx.relabel_nodes(mol_graph, mapping, copy=True)
                mol = Chem.RWMol()

                for node in mol_graph:
                    ase_atom = mol_graph.nodes[node]["atom"]
                    atom = Chem.Atom(ase_atom.symbol)
                    atom.SetDoubleProp("x", ase_atom.position[0])
                    atom.SetDoubleProp("y", ase_atom.position[1])
                    atom.SetDoubleProp("z", ase_atom.position[2])
                    mol.AddAtom(atom)
                for edge in mol_graph.edges:
                    mol.AddBond(int(edge[0]), int(edge[1]), Chem.BondType.SINGLE)

                molecules.append(mol)
        return molecules

    @property
    def atoms_fixed_interatomic_bonds(self) -> base.protocol.ATOMS_LST:
        atoms_lst = []
        for atoms, graph in zip(self.get_data(), self.graphs, strict=True):
            atoms.set_constraint(FixBondLengths(np.array(graph.edges)))
            atoms_lst.append(atoms)

        return atoms_lst

    def get_coarse_grained_atoms(self) -> base.protocol.ATOMS_LST:
        """Get coarse grained atoms by COM of molecules."""
        cc_atoms_list = []
        for graph, atoms in zip(self.graphs, self.data, strict=True):
            coms = []
            for mol_ids in nx.connected_components(graph):
                symbols = np.array(atoms.get_chemical_symbols())[list(mol_ids)]
                positions = np.array(atoms.get_positions())[list(mol_ids)]
                mol = ase.Atoms(
                    symbols=symbols,
                    positions=positions,
                    cell=atoms.get_cell(),
                    pbc=atoms.get_pbc(),
                )
                coms.append(mol.get_center_of_mass())
            cc_atoms_list.append(
                ase.Atoms(positions=coms, cell=atoms.get_cell(), pbc=atoms.get_pbc())
            )
        return cc_atoms_list

    def get_all_atoms(self, cc_atoms_list):
        """Get all atoms by COM of molecules.

        Parameters
        ----------
        cc_atoms_list : list of ase.Atoms
            List of coarse grained atoms.
            The atom ids and length must the same as the
            output generated by 'get_coarse_grained_atoms'.
            Positions and cell size can be different.

        Returns
        -------
        new_atoms : list of ase.Atoms
            List of all atoms based on the COM of the cc_atoms input.
            This will only affect the positions of the atoms.
            Rotations will be kept the same as in 'self.get_data()'.
        """
        new_atoms = []

        for graph, atoms, cc_atoms in zip(
            self.graphs, self.data, cc_atoms_list, strict=True
        ):
            new_symbols, new_positions = [], []
            for mol_ids, com in zip(
                nx.connected_components(graph), cc_atoms.get_positions(), strict=True
            ):
                symbols = np.array(atoms.get_chemical_symbols())[list(mol_ids)]
                positions = np.array(atoms.get_positions())[list(mol_ids)]
                mol = ase.Atoms(
                    symbols=symbols,
                    positions=positions,
                    cell=atoms.get_cell(),
                    pbc=atoms.get_pbc(),
                )
                mol.center(about=0)

                new_symbols.append(mol.get_chemical_symbols())
                new_positions.append(mol.get_positions() + com)

            new_atoms.append(
                ase.Atoms(
                    np.concatenate(new_symbols),
                    positions=np.concatenate(new_positions),
                    cell=cc_atoms.get_cell(),
                    pbc=cc_atoms.get_pbc(),
                )
            )
        return new_atoms
