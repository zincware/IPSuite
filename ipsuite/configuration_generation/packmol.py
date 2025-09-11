"""Use packmole to create a periodic box"""

import logging
import os
import pathlib
import random

import ase
import ase.units
import h5py
import numpy as np
import rdkit2ase
import znh5md
import zntrack

from ipsuite import base

log = logging.getLogger(__name__)


class Packmol(base.IPSNode):
    """Create a box with packmol.

    Attributes
    ----------
    data: list[list[ase.Atoms]]
        For each entry in the list the last ase.Atoms object is used to create the
        new structure.
    data_ids: list[int]
        The id of the data to use for each entry in data. If None the last entry.
        Has to be the same length as data. data: [[A], [B]], [-1, 3] -> [A[-1], B[3]]
    count: list[int]
        Number of molecules to add for each entry in data.
    tolerance : float
        Tolerance for the distance of atoms in angstrom.
    density : float
        Density of the system in kg/m^3. Either density or box is required.
    pbc : bool
        If True the periodic boundary conditions are set for the generated structure and
        the box used by packmol is scaled by the tolerance, to avoid overlapping atoms
        with periodic boundary conditions.

    Notes
    -----
    Output structures should be relaxed before further use.
    """

    data: list[list[ase.Atoms]] = zntrack.deps()
    data_ids: list[int] = zntrack.params(None)
    count: list = zntrack.params()
    tolerance: float = zntrack.params(2.0)
    density: float = zntrack.params()
    pbc: bool = zntrack.params(True)
    frames_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "frames.h5")

    def __post_init__(self):
        if len(self.data) != len(self.count):
            raise ValueError("The number of data and count must be the same.")

    def run(self):
        data = []
        if self.data_ids is not None:
            for idx, frames in zip(self.data_ids, self.data):
                data.append([frames[idx]])
        else:
            data = self.data

        frames = [
            rdkit2ase.pack(
                data=data,
                counts=self.count,
                tolerance=self.tolerance,
                density=self.density,
                pbc=self.pbc,
                packmol=os.environ.get("RDKIT2ASE_PACKMOL", "packmol"),
            )
        ]
        io = znh5md.IO(self.frames_path)
        io.extend(frames)

    @property
    def frames(self) -> list[ase.Atoms]:
        with self.state.fs.open(self.frames_path, "rb") as f:
            with h5py.File(f) as file:
                return znh5md.IO(file_handle=file)[:]


class MultiPackmol(Packmol):
    """Create multiple configurations with packmol.

    This Node generates multiple configurations with packmol.
    This is best used in conjunction with Smiles2Conformers:

    Attributes
    ----------
    n_configurations : int
        Number of configurations to create.
    seed : int
        Seed for the random number generator.

    Notes
    -----
    Output structures should be relaxed before further use.


    Example
    -------
    >>> import ipsuite as ips
    >>> project = ips.Project()
    >>> with project:
    ...     bf4 = ips.Smiles2Conformers(
    ...         smiles='[B-](F)(F)(F)F', numConfs=10
    ...     )
    ...     bmim = ips.Smiles2Conformers(
    ...         smiles='CCCCN1C=C[N+](=C1)C',
    ...         numConfs=10
    ...     )
    ...     molecules = ips.MultiPackmol(
    ...         data=[bf4.frames, bmim.frames],
    ...         count=[1, 1], density=1210, n_configurations=10
    ...     )
    ...     box = ips.MultiPackmol(
    ...         data=[molecules.frames], count=[10], density=1210, n_configurations=1
    ...     )
    >>> project.build()
    """

    n_configurations: int = zntrack.params()
    seed: int = zntrack.params(42)

    def run(self):
        np.random.seed(self.seed)
        frames = []
        for _ in range(self.n_configurations):
            # shuffle each data entry
            data = []
            for frame_list in self.data:
                random.shuffle(frame_list)
                data.append(frame_list)

            frames.append(
                rdkit2ase.pack(
                    data=data,
                    counts=self.count,
                    tolerance=self.tolerance,
                    density=self.density,
                    pbc=self.pbc,
                    packmol=os.environ.get("RDKIT2ASE_PACKMOL", "packmol"),
                )
            )
        io = znh5md.IO(self.frames_path)
        io.extend(frames)
