"""Use packmole to create a periodic box"""

import logging
import pathlib
import subprocess

import ase
import ase.units
import numpy as np
import zntrack
from ase.visualize import view

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
    box : list[float]
        Box size in angstrom. Either density or box is required.
    density : float
        Density of the system in kg/m^3. Either density or box is required.
    scale_box : bool
        If True the box used by packmol is scaled by the tolerance, to avoid
        overlapping atoms with periodic boundary conditions.
    pbc : bool
        If True the periodic boundary conditions are set for the generated structure.
    """

    data: list[list[ase.Atoms]] = zntrack.deps()
    data_ids: list[int] = zntrack.zn.params(None)
    count: list = zntrack.zn.params()
    tolerance: float = zntrack.zn.params(2.0)
    box: list = zntrack.zn.params(None)
    density: float = zntrack.zn.params(None)
    structures = zntrack.dvc.outs(zntrack.nwd / "packmol")
    atoms = fields.Atoms()
    scale_box: bool = zntrack.params(True)
    pbc: bool = zntrack.params(True)

    def _post_init_(self):
        if self.box is None and self.density is None:
            raise ValueError("Either box or density must be set.")
        if len(self.data) != len(self.count):
            raise ValueError("The number of data and count must be the same.")
        if self.data_ids is not None and len(self.data) != len(self.data_ids):
            raise ValueError("The number of data and data_ids must be the same.")
        if self.box is not None and isinstance(self.box, (int, float)):
            self.box = [self.box, self.box, self.box]

    def run(self):
        self.structures.mkdir(exist_ok=True, parents=True)
        for idx, atoms in enumerate(self.data):
            atoms = atoms[-1] if self.data_ids is None else atoms[self.data_ids[idx]]
            ase.io.write(self.structures / f"{idx}.xyz", atoms)

        if self.density is not None:
            self._get_box_from_molar_volume()

        if self.scale_box:
            scaled_box = [x - self.tolerance for x in self.box]
        else:
            scaled_box = self.box

        file = f"""
        tolerance {self.tolerance}
        filetype xyz
        output mixture.xyz
        """
        for idx, count in enumerate(self.count):
            file += f"""
            structure {idx}.xyz
                number {count}
                inside box 0 0 0 {" ".join([f"{x:.4f}" for x in scaled_box])}
            end structure
            """
        with pathlib.Path(self.structures / "packmole.inp").open("w") as f:
            f.write(file)

        subprocess.check_call("packmol < packmole.inp", shell=True, cwd=self.structures)

        atoms = ase.io.read(self.structures / "mixture.xyz")
        if self.pbc:
            atoms.cell = self.box
            atoms.pbc = True
        self.atoms = [atoms]

    def _get_box_from_molar_volume(self):
        """Get the box size from the molar volume"""
        molar_mass = [
            sum(atoms[0].get_masses()) * count
            for atoms, count in zip(self.data, self.count)
        ]
        molar_mass = sum(molar_mass)  #  g / mol
        molar_volume = molar_mass / self.density / 1000  # m^3 / mol

        # convert to particles / A^3
        volume = molar_volume * (ase.units.m**3) / ase.units.mol

        self.box = [volume ** (1 / 3) for _ in range(3)]
        log.info(f"estimated box size: {self.box}")

    def view(self) -> view:
        return view(self.atoms, viewer="x3d")


class MultiPackmol(Packmol):
    """Create multiple configurations with packmol.

    This Node generates multiple configurations with packmol.
    This is best used in conjunction with SmilesToConformers:

    >>> import ipsuite as ips
    >>> with ips.Project(automatic_node_names=True) as project:
    >>>    water = ips.configuration_generation.SmilesToConformers(
    >>>         smiles='O', numConfs=100
    >>>    )
    >>>    boxes = ips.configuration_generation.MultiPackmol(
    >>>         data=[water.atoms], count=[10], density=997, n_configurations=10
    >>>    )
    >>> project.run()

    Attributes
    ----------
    n_configurations : int
        Number of configurations to create.
    seed : int
        Seed for the random number generator.
    """

    n_configurations: int = zntrack.params()
    seed: int = zntrack.params(42)
    data_ids = None

    def run(self):
        np.random.seed(self.seed)
        self.atoms = []

        if self.density is not None:
            self._get_box_from_molar_volume()

        if self.scale_box:
            scaled_box = [x - self.tolerance for x in self.box]
        else:
            scaled_box = self.box

        self.structures.mkdir(exist_ok=True, parents=True)
        for idx, atoms_list in enumerate(self.data):
            for jdx, atoms in enumerate(atoms_list):
                ase.io.write(self.structures / f"{idx}_{jdx}.xyz", atoms)

        for idx in range(self.n_configurations):
            file = f"""
            tolerance {self.tolerance}
            filetype xyz
            output mixture_{idx}.xyz
            """
            for jdx, count in enumerate(self.count):
                choices = np.random.choice(len(self.data[jdx]), count)
                for kdx in choices:
                    file += f"""
                    structure {jdx}_{kdx}.xyz
                        number 1
                        inside box 0 0 0 {" ".join([f"{x:.4f}" for x in scaled_box])}
                    end structure
                    """
            with pathlib.Path(self.structures / f"packmole_{idx}.inp").open("w") as f:
                f.write(file)

            subprocess.check_call(
                f"packmol < packmole_{idx}.inp", shell=True, cwd=self.structures
            )

            atoms = ase.io.read(self.structures / f"mixture_{idx}.xyz")
            if self.pbc:
                atoms.cell = self.box
                atoms.pbc = True

            self.atoms.append(atoms)
