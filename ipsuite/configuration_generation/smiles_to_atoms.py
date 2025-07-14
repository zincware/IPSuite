from pathlib import Path

import ase
import h5py
import rdkit2ase
import znh5md
import zntrack
from rdkit.Chem import Draw

from ipsuite import base


class Smiles2Atoms(base.IPSNode):
    smiles: str = zntrack.params()
    seed: int = zntrack.params(1234)

    frames_path: Path = zntrack.outs_path(zntrack.nwd / "frames.h5")
    molecule_image_path: Path = zntrack.outs_path(zntrack.nwd / "molecule.png")

    def run(self):
        atoms = rdkit2ase.smiles2atoms(
            smiles=self.smiles,
            seed=self.seed,
        )

        io = znh5md.IO(filename=self.frames_path)
        io.append(atoms)

        # Generate and save molecule image
        mol = rdkit2ase.ase2rdkit(atoms)
        img = Draw.MolToImage(mol, size=(300, 300))
        img.save(self.molecule_image_path)

    @property
    def frames(self) -> list[ase.Atoms]:
        with self.state.fs.open(self.frames_path, "rb") as f:
            with h5py.File(f) as file:
                return znh5md.IO(file_handle=file)[:]


class Smiles2Conformers(base.IPSNode):
    smiles: str = zntrack.params()
    numConfs: int = zntrack.params()
    seed: int = zntrack.params(42)
    maxAttempts: int = zntrack.params(1000)

    frames_path: Path = zntrack.outs_path(zntrack.nwd / "frames.h5")
    molecule_image_path: Path = zntrack.outs_path(zntrack.nwd / "molecule.png")

    def run(self):
        frames = rdkit2ase.smiles2conformers(
            smiles=self.smiles,
            numConfs=self.numConfs,
            randomSeed=self.seed,
            maxAttempts=self.maxAttempts,
        )

        io = znh5md.IO(filename=self.frames_path)
        io.extend(frames)

        # Generate and save molecule image using first conformer
        if frames:
            mol = rdkit2ase.ase2rdkit(frames[0])
            img = Draw.MolToImage(mol, size=(300, 300))
            img.save(self.molecule_image_path)

    @property
    def frames(self) -> list[ase.Atoms]:
        with self.state.fs.open(self.frames_path, "rb") as f:
            with h5py.File(f) as file:
                return znh5md.IO(file_handle=file)[:]
