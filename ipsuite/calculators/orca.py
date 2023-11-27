import tqdm
import znh5md
import zntrack
from ase.calculators.orca import ORCA

from ipsuite import base


class OrcaSinglePoint(base.ProcessAtoms):
    orcasimpleinput: str = zntrack.zn.params("B3LYP def2-TZVP")
    orcablocks: str = zntrack.zn.params("%pal nprocs 16 end")
    ASE_ORCA_COMMAND: str = zntrack.meta.Environment("orca")

    orca_directory: str = zntrack.dvc.outs(zntrack.nwd / "orca")
    output_file: str = zntrack.dvc.outs(zntrack.nwd / "atoms.h5")

    def run(self):
        db = znh5md.io.DataWriter(self.output_file)
        db.initialize_database_groups()

        calc = self.get_calculator()
        for atoms in tqdm.tqdm(self.get_data(), ncols=70):
            atoms.calc = calc
            atoms.get_potential_energy()
            db.add(znh5md.io.AtomsReader([atoms]))

    @property
    def atoms(self):
        return znh5md.ASEH5MD(self.output_file).get_atoms_list()

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
