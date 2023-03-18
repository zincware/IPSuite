"""Molecule Mapping using smiles and networkx"""


import typing

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

    # TODO remove n_molecules
    # TODO make smiles a dict {ratio: smiles}
    #  where ratio can be used e.g. MgCl2 would be {1: "Mg", 2: "Cl"}

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
            for idx, _ in enumerate(atoms):
                graph.add_node(idx)

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

        self.atoms = self.get_coarse_grained_atoms()

    def get_rdkit_molecules(self, item=None):
        if item is None:
            item = slice(None)
        molecules = []

        for graph, ase_atoms in zip(
            self.graphs[item], self.get_data()[item], strict=True
        ):
            for mol_ids in nx.connected_components(graph):
                mol_graph = graph.subgraph(mol_ids)
                mol = Chem.RWMol()

                for node in mol_graph:
                    ase_atom = ase_atoms[node]
                    atom = Chem.Atom(ase_atom.symbol)
                    atom.SetDoubleProp("x", ase_atom.position[0])
                    atom.SetDoubleProp("y", ase_atom.position[1])
                    atom.SetDoubleProp("z", ase_atom.position[2])
                    mol.AddAtom(atom)

                mapping = {idx: val for val, idx in enumerate(mol_graph.nodes)}
                mol_graph_relabled = nx.relabel_nodes(mol_graph, mapping, copy=True)
                for edge in mol_graph_relabled.edges:
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

    def get_molecule_ids(self, graph) -> typing.List[typing.List[int]]:
        """Get molecule ids from a graph."""
        return [list(mol_ids) for mol_ids in nx.connected_components(graph)]

    def get_com(self, atoms: ase.Atoms) -> np.ndarray:
        """Get center of mass of atoms.

        Try to accoung for pbc.
        """
        atoms = atoms.copy()
        reference = atoms.get_positions(wrap=True)[0]
        atoms.center()
        atoms.wrap()

        shift = atoms.get_positions(wrap=True)[0] - reference
        return atoms.get_center_of_mass() - shift

    def get_coarse_grained_atoms(self) -> base.protocol.ATOMS_LST:
        """Get coarse grained atoms by COM of molecules."""
        cc_atoms_list = []
        for graph, atoms in zip(self.graphs, self.get_data(), strict=True):
            coms = []
            for mol_ids in self.get_molecule_ids(graph):
                atomic_numbers = np.array(atoms.get_atomic_numbers())[mol_ids]
                positions = np.array(atoms.get_positions())[mol_ids]
                mol = ase.Atoms(
                    atomic_numbers,
                    positions=positions,
                    cell=atoms.get_cell(),
                    pbc=atoms.get_pbc(),
                )
                coms.append(self.get_com(mol))

            cc_atoms_list.append(
                ase.Atoms(positions=coms, cell=atoms.get_cell(), pbc=atoms.get_pbc())
            )
        return cc_atoms_list

    def get_all_atoms(self, data):
        """Get all atoms by COM of molecules.

        Parameters
        ----------
        data : list|dict of ase.Atoms
            List or dict of coarse grained atoms.
            if provides as list:
                The atom ids and length must the same as the
                output generated by 'get_coarse_grained_atoms'.
            if provided as dict:
                the keys must be the 0based index of the atoms
                in the output generated by 'get_coarse_grained_atoms'.
            Positions and cell size can be different.

        Returns
        -------
        new_atoms : list of ase.Atoms
            List of all atoms based on the COM of the data input.
            This will only affect the positions of the atoms.
            Rotations will be kept the same as in 'self.get_data()'.
        """

        # TODO a simple test: coarse grain and then undo and check if the same
        #  currently distances in pbc are the same but not the positions
        new_atoms = []

        if isinstance(data, list):
            data = dict(enumerate(data))

        self.update_data()

        for idx in data:
            graph = self.graphs[idx]
            atoms = self.data[idx]
            cc_atoms = data[idx]

            new_atomic_numbers, new_positions = [], []
            for mol_ids, com in zip(
                self.get_molecule_ids(graph), cc_atoms.get_positions(), strict=True
            ):
                atomic_numbers = np.array(atoms.get_atomic_numbers())[mol_ids]
                positions = np.array(atoms.get_positions())[mol_ids]
                mol = ase.Atoms(
                    atomic_numbers,
                    positions=positions,
                    cell=atoms.get_cell(),
                    pbc=atoms.get_pbc(),
                )
                mol.positions -= self.get_com(mol)
                # mol.wrap() # TODO this is not working

                new_atomic_numbers.append(mol.get_atomic_numbers())
                new_positions.append(mol.get_positions() + com)

            new_atoms.append(
                ase.Atoms(
                    np.concatenate(new_atomic_numbers),
                    positions=np.concatenate(new_positions),
                    cell=cc_atoms.get_cell(),
                    pbc=cc_atoms.get_pbc(),
                )
            )
            new_atoms[-1].wrap()
        return new_atoms
