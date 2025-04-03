import contextlib
import os
import pathlib

import ase
import h5py
import tqdm
import znh5md
import zntrack
from ase.calculators.orca import ORCA, OrcaProfile

from ipsuite import base


class OrcaSinglePoint(base.IPSNode):
    """Node for running ORCA Single point calculations.

    Parameters
    ----------
    orca_shell : str, default=None
        The cmd to run orca. If None, the environment variable
        IPSUITE_ORCA_SHELL is used.
    """

    data: list[ase.Atoms] = zntrack.deps()

    orcasimpleinput: str = zntrack.params("B3LYP def2-TZVP")
    orcablocks: str = zntrack.params("%pal nprocs 16 end")
    orca_shell: str | None = zntrack.params(None)

    orca_directory: pathlib.Path = zntrack.outs_path(zntrack.nwd / "orca")
    output_file: str = zntrack.outs_path(zntrack.nwd / "structures.h5")

    def run(self):
        db = znh5md.IO(self.output_file)

        skip = 0
        with contextlib.suppress(FileNotFoundError):
            skip = len(db)

        calc = self.get_calculator()
        for idx, atoms in enumerate(tqdm.tqdm(self.data, ncols=70, total=len(self.data))):
            if idx < skip:
                continue
            atoms.calc = calc
            atoms.get_potential_energy()
            db.append(atoms)

    @property
    def frames(self) -> list[ase.Atoms]:
        with self.state.fs.open(self.output_file, "rb") as f:
            with h5py.File(f) as file:
                return znh5md.IO(file_handle=file)[:]

    def get_calculator(self, directory: str = None):
        if directory is None:
            directory = self.orca_directory

        orca_shell = os.environ.get("IPSUITE_ORCA_SHELL", self.orca_shell)

        profile = OrcaProfile(command=orca_shell)

        calc = ORCA(
            profile=profile,
            orcasimpleinput=self.orcasimpleinput,
            orcablocks=self.orcablocks,
            directory=directory,
        )
        return calc
