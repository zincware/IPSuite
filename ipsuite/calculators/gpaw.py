import functools
import pathlib
import typing
import ase

import tqdm
import znh5md
import zntrack
import h5py

from ipsuite import base

import gpaw


class GPAWSinglePoint(base.ProcessAtoms):
    """Rudimentary GPAW interface.
    Currently only supports energies and forces in PW mode.
    See the GPAW documentation for an explanation of the parameters.
    """

    pw: int = zntrack.params(200)
    xc: str = zntrack.params("PBE")
    kpts: tuple[int]= zntrack.params((4,4,4))
    eigensolver: str = zntrack.params("rmm-diis")
    occupations: float = zntrack.params(0.0)
    hund: bool = zntrack.params(False)

    output_file: pathlib.Path = zntrack.outs_path(zntrack.nwd / "atoms.h5")
    gpaw_directory: pathlib.Path = zntrack.outs_path(zntrack.nwd / "gpaw")


    def run(self):
        if not self.gpaw_directory.exists():
            self.gpaw_directory.mkdir(exist_ok=True)

        db = znh5md.IO(self.output_file)

        calc = self.get_calculator()

        for atoms in tqdm.tqdm(self.get_data()):
            atoms.calc = calc
            atoms.get_potential_energy()
            atoms.get_forces()
            db.append(atoms)

    @property
    def atoms(self) -> typing.List[ase.Atoms]:
        with self.state.fs.open(self.output_file, "rb") as f:
            with h5py.File(f) as file:
                return znh5md.IO(file_handle=file)[:]


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
            txt=(directory / "gpaw.out").as_posix(),
        )
        return calc
