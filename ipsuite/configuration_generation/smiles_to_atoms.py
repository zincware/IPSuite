import ase
import rdkit2ase
import zntrack

from ipsuite import base, fields


class Smiles2Atoms(base.IPSNode):
    frames: list[ase.Atoms] = fields.Atoms()

    smiles: str = zntrack.params()
    cell: float = zntrack.params(None)
    seed: int = zntrack.params(1234)

    def run(self):
        atoms = rdkit2ase.smiles2atoms(
            smiles=self.smiles,
            seed=self.seed,
        )
        if self.cell:
            atoms.set_cell([self.cell, self.cell, self.cell])
            atoms.center()
        self.frames = [atoms]


class Smiles2Conformers(base.IPSNode):
    frames: list[ase.Atoms] = fields.Atoms()

    smiles: str = zntrack.params()
    numConfs: int = zntrack.params()
    seed: int = zntrack.params(42)
    maxAttempts: int = zntrack.params(1000)
    cell: float = zntrack.params(100)

    def run(self):
        frames = rdkit2ase.smiles2conformers(
            smiles=self.smiles,
            numConfs=self.numConfs,
            randomSeed=self.seed,
            maxAttempts=self.maxAttempts,
        )
        if self.cell:
            for atoms in frames:
                atoms.set_cell([self.cell, self.cell, self.cell])
                atoms.center()
        self.frames = frames
