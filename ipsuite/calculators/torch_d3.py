import contextlib

import ase
import numpy as np
import torch
import tqdm
import zntrack
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.singlepoint import SinglePointCalculator
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

from ipsuite import base, fields
from ipsuite.utils.ase_sim import freeze_copy_atoms


class TorchD3(base.ProcessAtoms):
    """Compute D3 correction terms using torch-dftd.

    Attributes
    ----------
    xc : str
    damping : str
    cutoff : float
    abc : bool
        ATM 3-body interaction
    cnthr : float
        Coordination number cutoff distance in angstrom
    dtype : str
        Data type used for the calculation.
    device : str
        Device used for the calculation. Defaults to "cuda" if available, otherwise "cpu".
    """

    xc: str = zntrack.params()
    damping: str = zntrack.params()
    cutoff: float = zntrack.params()
    abc: bool = zntrack.params()
    cnthr: float = zntrack.params()
    dtype: str = zntrack.params()
    device: str = zntrack.meta.Text(None)

    atoms: list[ase.Atoms] = fields.Atoms()

    def _post_load_(self) -> None:
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(self) -> None:
        calc = self.get_calculator()
        self.atoms = []
        for atoms in tqdm.tqdm(self.get_data(), ncols=70):

            if atoms.calc is None:
                atoms.calc = SinglePointCalculator(
                    atoms,
                    energy=0,
                    forces=np.zeros_like(atoms.positions),
                    stress=np.zeros((3, 3)),
                )

            _atoms = freeze_copy_atoms(atoms)

            atoms.calc = calc
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()

            with contextlib.suppress(KeyError):
                _atoms.calc.results["energy"] += energy
            with contextlib.suppress(KeyError):
                _atoms.calc.results["forces"] += forces

            with contextlib.suppress(PropertyNotImplementedError):
                # non periodic systems
                stress = atoms.get_stress()
                with contextlib.suppress(KeyError):
                    _atoms.calc.results["stress"] += stress

            self.atoms.append(_atoms)

    def get_calculator(self, **kwargs):
        if self.dtype == "float64":
            dtype = torch.float64
        elif self.dtype == "float32":
            dtype = torch.float32
        else:
            raise ValueError("dtype must be float64 or float32")

        return TorchDFTD3Calculator(
            xc=self.xc,
            damping=self.damping,
            cutoff=self.cutoff,
            abc=self.abc,
            cnthr=self.cnthr,
            dtype=dtype,
            atoms=None,
            device=self.device,
        )
