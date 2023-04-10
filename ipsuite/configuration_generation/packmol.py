"""Use packmole to create a periodic box"""
import pathlib
import subprocess

import ase
import zntrack
from ase.visualize import view


class Packmol(zntrack.Node):
    """

    Attributes
    ----------

    """

    data: list = zntrack.zn.deps()
    count: list = zntrack.zn.params()
    tolerance: float = zntrack.zn.params(2.0)
    box: list = zntrack.zn.params()
    structures = zntrack.dvc.outs(zntrack.nwd)

    def _post_init_(self):
        if len(self.data) != len(self.count):
            raise ValueError("The number of data and count must be the same.")
        if isinstance(self.box, (int, float)):
            self.box = [self.box, self.box, self.box]

    def run(self):
        for idx, atoms in enumerate(self.data):
            ase.io.write(self.structures / f"{idx}.xyz", atoms)

        file = f"""
        tolerance {self.tolerance}
        filetype xyz
        output mixture.xyz
        """
        for idx, count in enumerate(self.count):
            file += f"""
            structure {idx}.xyz
                number {count}
                inside box 0 0 0 {" ".join([str(x) for x in self.box])}
            end structure
            """
        with pathlib.Path(self.structures / "packmole.inp").open("w") as f:
            f.write(file)

        subprocess.check_call("packmol < packmole.inp", shell=True, cwd=self.structures)

    @property
    def atoms(self):
        atoms = ase.io.read(self.structures / "mixture.xyz")
        atoms.cell = self.box
        atoms.pbc = True
        return atoms

    def view(self) -> view:
        return view(self.atoms, viewer="x3d")
