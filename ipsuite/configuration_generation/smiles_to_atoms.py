from pathlib import Path

import ase
import h5py
import rdkit2ase
import znh5md
import zntrack
import contextlib

from ipsuite import base, fields
from ipsuite.utils.helpers import make_hdf5_file_opener


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
    def frames(self) -> znh5md.IO:
        file_factory = make_hdf5_file_opener(self, self.frames_path)
        return znh5md.IO(file_factory=file_factory)
