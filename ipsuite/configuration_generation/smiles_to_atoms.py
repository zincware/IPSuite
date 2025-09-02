from pathlib import Path

import ase
import h5py
import rdkit2ase
import typing_extensions as tyex
import znh5md
import zntrack
from rdkit.Chem import Draw

from ipsuite import base


@tyex.deprecated("Use `ipsuite.Smiles2Conformers` instead.")
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
    """Generate molecular conformers from a SMILES string.

    Attributes
    ----------
    smiles : str
        The SMILES string representing the molecule.
    numConfs : int
        The number of conformers to generate.
    seed : int, optional
        Random seed for conformer generation (default is 42).
    maxAttempts : int, optional
        Maximum number of attempts to generate conformers (default is 1000).

    Methods
    -------
    frames : list of ase.Atoms
        Property to load and return the generated conformers as a list of ASE Atoms.

    Notes
    -----
    Instead of creating one composite smile like `[B-](F)(F)(F)F.CCCCN1C=C[N+](=C1)C`
    create two molecules and use `MultiPackmol` to generate the single molecule.
    This will avoid overlapping structures.

    Examples
    --------
    >>> with project:
    ...     methanol_conformers = ips.Smiles2Conformers(smiles="CO", numConfs=5)
    >>> project.repro()
    >>> frames = methanol_conformers.frames
    >>> print(f"Generated {len(frames)} conformers.")
    Generated 5 conformers.
    """

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
