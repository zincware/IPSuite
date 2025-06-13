import ase
import rdkit2ase
import zntrack
from pathlib import Path

from ipsuite import base, fields
import znh5md
import h5py


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
    smiles: str = zntrack.params()
    numConfs: int = zntrack.params()
    seed: int = zntrack.params(42)
    maxAttempts: int = zntrack.params(1000)
    cell: float = zntrack.params(100)

    frames_path: Path = zntrack.outs_path(zntrack.nwd / "frames.h5")

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
        
        io = znh5md.IO(filename=self.frames_path)
        io.extend(frames)

    @property
    def frames(self) -> list[ase.Atoms]:
        with self.state.fs.open(self.frames_path, "rb") as f:
            with h5py.File(f) as file:
                return znh5md.IO(file_handle=file)[:]