"""Use packmole to create a periodic box"""

import logging
import os
import random

import ase
import ase.units
import numpy as np
import rdkit2ase
import zntrack

from ipsuite import base, fields

log = logging.getLogger(__name__)


class Packmol(base.IPSNode):
    """

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
    """

    data: list[list[ase.Atoms]] = zntrack.deps()
    data_ids: list[int] = zntrack.params(None)
    count: list = zntrack.params()
    tolerance: float = zntrack.params(2.0)
    density: float = zntrack.params()
    frames: list[ase.Atoms] = fields.Atoms()
    pbc: bool = zntrack.params(True)

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

        self.frames = [
            rdkit2ase.pack(
                data=data,
                counts=self.count,
                tolerance=self.tolerance,
                density=self.density,
                pbc=self.pbc,
                verbose=bool(os.environ.get("IPSUITE_PACKMOL_VERBOSE", False)),
            )
        ]


class MultiPackmol(Packmol):
    """Create multiple configurations with packmol.

    This Node generates multiple configurations with packmol.
    This is best used in conjunction with Smiles2Conformers:

    Example
    -------
    .. testsetup::
        >>> tmp_path = utils.docs.create_dvc_git_env_for_doctest()

    >>> import ipsuite as ips
    >>> with ips.Project() as project:
    ...     water = ips.Smiles2Conformers(
    ...         smiles='O', numConfs=100
    ...         )
    ...     boxes = ips.MultiPackmol(
    ...         data=[water.frames], count=[10], density=997, n_configurations=10
    ...         )
    >>> project.repro()

    .. testcleanup::
        >>> tmp_path.cleanup()

    Attributes
    ----------
    n_configurations : int
        Number of configurations to create.
    seed : int
        Seed for the random number generator.
    """

    n_configurations: int = zntrack.params()
    seed: int = zntrack.params(42)

    def run(self):
        np.random.seed(self.seed)
        self.frames = []
        for _ in range(self.n_configurations):
            # shuffle each data entry
            data = []
            for frames in self.data:
                random.shuffle(frames)
                data.append(frames)

            self.frames.append(
                rdkit2ase.pack(
                    data=data,
                    counts=self.count,
                    tolerance=self.tolerance,
                    density=self.density,
                    pbc=self.pbc,
                    verbose=bool(os.environ.get("IPSUITE_PACKMOL_VERBOSE", False)),
                )
            )
