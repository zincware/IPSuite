import os
import pathlib
import typing

import tqdm
import zntrack
from ase.calculators.subprocesscalculator import gpaw_process
from gpaw.eigensolvers.eigensolver import Eigensolver
from gpaw.external import ExternalPotential
from gpaw.poisson import _PoissonSolver
from gpaw.wavefunctions.mode import Mode

from ipsuite import base
from ipsuite.utils.ase_sim import freeze_copy_atoms


class GPAWSinglePoint(base.ProcessAtoms):
    """GPAW single-point calculation interface. Currently only supports
    energies and forces. See the GPAW documentation for an explanation of the
    parameters: https://gpaw.readthedocs.io/documentation/basic.html
    """

    mode: str | dict | Mode = zntrack.params(None)
    xc: str = zntrack.params("PBE")
    occupations: dict = zntrack.params(None)
    poissonsolver: dict | _PoissonSolver = zntrack.params(None)
    h: float = zntrack.params(None)
    gpts: tuple[int, int, int] = zntrack.params(None)
    kpts: dict | tuple[int, int, int] = zntrack.params(None)
    nbands: int | str = zntrack.params(None)
    charge: float = zntrack.params(0.0)
    setups: str | dict = zntrack.params("paw")
    basis: str | dict = zntrack.params(None)
    spinpol: bool = zntrack.params(None)
    mixer: dict = zntrack.params(None)
    eigensolver: str | Eigensolver = zntrack.params("rmm-diis")
    external: ExternalPotential = zntrack.params(None)
    random: bool = zntrack.params(False)
    hund: bool = zntrack.params(False)
    maxiter: int = zntrack.params(333)
    symmetry: dict | str = zntrack.params(None)
    convergence: dict = zntrack.params({})

    ncores: int = zntrack.params(None)
    gpaw_directory: pathlib.Path = zntrack.outs_path(zntrack.nwd / "gpaw")

    def run(self):
        if not self.gpaw_directory.exists():
            self.gpaw_directory.mkdir(exist_ok=True)

        self.atoms = []
        with self.get_calculator() as calc:
            for atoms in tqdm.tqdm(self.get_data()):
                atoms.calc = calc
                atoms.get_potential_energy()
                atoms.get_forces()
                self.atoms.append(freeze_copy_atoms(atoms))

    def get_calculator(self, directory: str = None):
        if directory is None:
            directory = self.gpaw_directory
        else:
            directory = pathlib.Path(directory)

        if self.ncores is None:
            ncores = len(os.sched_getaffinity(0))
        else:
            ncores = self.ncores

        calc = gpaw_process(
            ncores=ncores,
            txt=str(directory / "gpaw.out"),
            **self._get_calculator_kwargs()
        )
        return calc

    def _get_calculator_kwargs(self) -> dict[str, typing.Any]:
        kwargs = dict(
            mode=self.mode,
            xc=self.xc,
            occupations=self.occupations,
            poissonsolver=self.poissonsolver,
            h=self.h,
            gpts=self.gpts,
            kpts=self.kpts,
            nbands=self.nbands,
            charge=self.charge,
            setups=self.setups,
            basis=self.basis,
            spinpol=self.spinpol,
            mixer=self.mixer,
            eigensolver=self.eigensolver,
            external=self.external,
            random=self.random,
            hund=self.hund,
            maxiter=self.maxiter,
            symmetry=self.symmetry,
            convergence=self.convergence
        )

        # let GPAW's initialization handle setting defaults
        return {k: v for k, v in kwargs.items() if v is not None}
