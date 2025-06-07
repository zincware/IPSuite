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
    """
    Generate molecular conformers from a SMILES string
    and store them as ASE Atoms objects.

    Parameters
    ----------
    smiles : str
        The SMILES string representing the molecular structure.
    numConfs : int, optional
        The number of conformers to generate (default is 1).
    seed : int, optional
        Random seed for conformer generation (default is 42).
    maxAttempts : int, optional
        Maximum number of attempts to generate conformers (default is 1000).
    cell : float, optional
        Size of the cubic unit cell to assign to the generated 
        conformers (default is 100).

    Attributes
    ----------
    frames : list[ase.Atoms]
        A list of generated molecular conformers as ASE Atoms objects.
    """

    frames: list[ase.Atoms] = fields.Atoms()

    smiles: str = zntrack.params()
    numConfs: int = zntrack.params(1)
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
