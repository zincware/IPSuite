"""Use packmole to create a periodic box"""
import logging
import pathlib
import subprocess

import ase
import ase.units
import zntrack
from ase.visualize import view

log = logging.getLogger(__name__)


class Packmol(zntrack.Node):
    """

    Attributes
    ----------
    density : float
        Density of the mixture in kg/m^3

    """

    data: list = zntrack.zn.deps()
    count: list = zntrack.zn.params()
    tolerance: float = zntrack.zn.params(2.0)
    box: list = zntrack.zn.params(None)
    density: float = zntrack.zn.params(None)
    structures = zntrack.dvc.outs(zntrack.nwd)

    def _post_init_(self):
        if self.box is None and self.density is None:
            raise ValueError("Either box or density must be set.")
        if len(self.data) != len(self.count):
            raise ValueError("The number of data and count must be the same.")
        if self.box is not None and isinstance(self.box, (int, float)):
            self.box = [self.box, self.box, self.box]

    def run(self):
        for idx, atoms in enumerate(self.data):
            ase.io.write(self.structures / f"{idx}.xyz", atoms)

        if self.density is not None:
            self._get_box_from_molar_volume()
        file = f"""
        tolerance {self.tolerance}
        filetype xyz
        output mixture.xyz
        """
        for idx, count in enumerate(self.count):
            file += f"""
            structure {idx}.xyz
                number {count}
                inside box 0 0 0 {" ".join([f"{x:.4f}" for x in self.box])}
            end structure
            """
        with pathlib.Path(self.structures / "packmole.inp").open("w") as f:
            f.write(file)

        subprocess.check_call("packmol < packmole.inp", shell=True, cwd=self.structures)

    def _get_box_from_molar_volume(self):
        """Get the box size from the molar volume"""
        molar_mass = [
            sum(atoms[0].get_masses()) * count
            for atoms, count in zip(self.data, self.count)
        ]
        molar_mass = sum(molar_mass)  #  g / mol
        molar_volume = molar_mass / 626 / 1000  # m^3 / mol

        # convert to particles / A^3
        volume = molar_volume * (ase.units.m**3) / ase.units.mol

        self.box = [volume ** (1 / 3) for _ in range(3)]
        log.info(f"estimated box size: {self.box}")

    @property
    def atoms(self):
        if self.density is not None:
            self._get_box_from_molar_volume()
        atoms = ase.io.read(self.structures / "mixture.xyz")
        atoms.cell = self.box
        atoms.pbc = True
        return atoms

    def view(self) -> view:
        return view(self.atoms, viewer="x3d")
