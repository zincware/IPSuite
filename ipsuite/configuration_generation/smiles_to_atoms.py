import pathlib

import ase
import zntrack
from ase.visualize import view
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from ipsuite import base, fields


class SmilesToAtoms(base.IPSNode):
    atoms: list[ase.Atoms] = fields.Atoms()

    smiles: str = zntrack.params()
    cell: float = zntrack.params(None)
    seed: int = zntrack.params(1234)
    optimizer: str = zntrack.params("UFF")
    image: pathlib.Path = zntrack.outs_path(zntrack.nwd / "molecule.png")

    def run(self):
        mol = Chem.MolFromSmiles(self.smiles)
        Draw.MolToFile(mol, self.image)

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=self.seed)

        if self.optimizer == "UFF":
            AllChem.UFFOptimizeMolecule(mol)
        elif self.optimizer == "MMFF":
            AllChem.MMFFOptimizeMolecule(mol)

        atoms = ase.Atoms(
            positions=mol.GetConformer().GetPositions(),
            numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()],
        )
        atoms.positions -= atoms.get_center_of_mass()
        if self.cell is not None:
            atoms.set_cell([self.cell, self.cell, self.cell])
            atoms.center()
        self.atoms = [atoms]

    def view(self) -> view:
        return view(self.atoms[0], viewer="x3d")


class SmilesToConformers(base.IPSNode):
    frames: list[ase.Atoms] = fields.Atoms()

    smiles: str = zntrack.params()
    numConfs: int = zntrack.params()
    seed: int = zntrack.params(42)
    maxAttempts: int = zntrack.params(1000)
    cell: float = zntrack.params(100)

    def run(self):
        mol = Chem.MolFromSmiles(self.smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(
            mol,
            numConfs=self.numConfs,
            randomSeed=self.seed,
            maxAttempts=self.maxAttempts,
        )
        self.frames = []
        for conf in mol.GetConformers():
            atoms = ase.Atoms(
                positions=conf.GetPositions(),
                numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()],
            )
            atoms.positions -= atoms.get_center_of_mass()

            if self.cell is not None:
                atoms.set_cell([self.cell, self.cell, self.cell])
                atoms.center()

            self.frames.append(atoms)
