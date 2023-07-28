import functools
import pathlib
import typing
import ase

import gpaw
import tqdm
import znh5md
import zntrack
import h5py

from ipsuite import base




class GPAWSinglePoint(base.ProcessAtoms):
    """Rudimentary GPAW interface.
    Currently only supports energies and forces in PW mode.
    See the GPAW documentation for an explanation of the parameters.
    """

    pw: int = zntrack.zn.params(200)
    xc: str = zntrack.zn.params("PBE")
    kpts: tuple[int]= zntrack.zn.params((4,4,4))
    eigensolver: str = zntrack.zn.params("rmm-diis")
    occupations: float = zntrack.zn.params(0.0)
    hund: bool = zntrack.zn.params(False)

    output_file: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "atoms.h5")
    gpaw_directory: pathlib.Path = zntrack.dvc.outs(zntrack.nwd)


    def run(self):
        db = znh5md.io.DataWriter(self.output_file)
        db.initialize_database_groups()

        calc = self.get_calculator()

        for atoms in tqdm.tqdm(self.get_data()):
            atoms.calc = calc
            atoms.get_potential_energy()
            atoms.get_forces()
            db.add(znh5md.io.AtomsReader([atoms]))

    @property
    def atoms(self) -> typing.List[ase.Atoms]:
        def file_handle(filename):
            file = self.state.fs.open(filename, "rb")
            return h5py.File(file)

        return znh5md.ASEH5MD(
            self.output_file,
            format_handler=functools.partial(
                znh5md.FormatHandler, file_handle=file_handle
            ),
        ).get_atoms_list()


    def get_calculator(self, directory: str = None):
        if directory is None:
            directory = self.gpaw_directory
        else:
            directory = pathlib.Path(directory)

        calc = gpaw.GPAW(
            mode=gpaw.PW(self.pw),
            xc=self.xc,
            hund=self.hund,
            kpts=self.kpts,
            eigensolver=self.eigensolver,
            occupations=gpaw.FermiDirac(self.occupations, fixmagmom=True),
            txt=directory / "gpaw.out",
        )
        return calc
