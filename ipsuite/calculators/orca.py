import h5py
import tqdm
import znh5md
import zntrack
from ase.calculators.orca import ORCA

from ipsuite import base


class OrcaSinglePoint(base.ProcessAtoms):
    orcasimpleinput: str = zntrack.params("B3LYP def2-TZVP")
    orcablocks: str = zntrack.params("%pal nprocs 16 end")
    ASE_ORCA_COMMAND: str = zntrack.meta.Environment("orca")

    orca_directory: str = zntrack.outs_path(zntrack.nwd / "orca")
    output_file: str = zntrack.outs_path(zntrack.nwd / "structures.h5")

    def run(self):
        db = znh5md.IO(self.output_file)

        calc = self.get_calculator()
        for atoms in tqdm.tqdm(self.get_data(), ncols=70):
            atoms.calc = calc
            atoms.get_potential_energy()
            db.append(atoms)

    @property
    def atoms(self):
        with self.state.fs.open(self.output_file, "rb") as f:
            with h5py.File(f) as file:
                return znh5md.IO(file_handle=file)[:]

    def get_calculator(self, directory: str = None):
        if directory is None:
            directory = self.orca_directory

        calc = ORCA(
            label="orcacalc",
            orcasimpleinput=self.orcasimpleinput,
            orcablocks=self.orcablocks,
            directory=directory,
            command=f"{self.ASE_ORCA_COMMAND} PREFIX.inp > PREFIX.out",
        )
        return calc
