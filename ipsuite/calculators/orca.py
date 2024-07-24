import tqdm
import znh5md
import zntrack
from ase.calculators.orca import ORCA, OrcaProfile
import pathlib

from ipsuite import base


class OrcaSinglePoint(base.ProcessAtoms):
    orcasimpleinput: str = zntrack.params("B3LYP def2-TZVP")
    orcablocks: str = zntrack.params("%pal nprocs 16 end")
    ASE_ORCA_COMMAND: str = zntrack.meta.Environment("orca")

    orca_directory: pathlib.Path = zntrack.outs_path(zntrack.nwd / "orca")
    output_file: str = zntrack.outs_path(zntrack.nwd / "structures.h5")

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
            
        profile = OrcaProfile(command=self.ASE_ORCA_COMMAND)
        
        calc = ORCA(
            profile=profile,
            orcasimpleinput=self.orcasimpleinput,
            orcablocks=self.orcablocks,
            directory=directory,
        )
        return calc
