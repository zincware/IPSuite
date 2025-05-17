import logging

import tqdm
import typing_extensions as tyex
import zntrack

from ipsuite import base
from ipsuite.utils.ase_sim import freeze_copy_atoms

log = logging.getLogger(__name__)


@tyex.deprecated(
    "Use `ipsuite.TBLiteModel` instead. Reason: Replaced by off-graph implementation."
)
class xTBSinglePoint(base.ProcessAtoms):
    """Node for labeling date with xTB and obtaining ASE calculators.

    Installation:
    conda install conda-forge::tblite-python

    Attributes
    ----------
    method: str
        xTB method to be used. Only "GFN1-xTB" supports PBC.
    """

    method: str = zntrack.params("GFN1-xTB")
    charge: int = zntrack.params(None)
    multiplicity: int = zntrack.params(None)
    accuracy: float = zntrack.params(1.0)
    electronic_temperature: float = zntrack.params(300.0)
    max_iterations: int = zntrack.params(250)
    initial_guess: str = zntrack.params("sad")
    mixer_damping: float = zntrack.params(0.4)
    spin_polarization: float = zntrack.params(None)

    def run(self):
        self.frames = []

        calculator = self.get_calculator()

        for atom in tqdm.tqdm(self.get_data(), ncols=70):
            atom.calc = calculator
            atom.get_potential_energy()
            self.frames.append(freeze_copy_atoms(atom))

    def get_calculator(self, **kwargs):
        """Get an xtb ase calculator."""
        try:
            from tblite.ase import TBLite
        except ImportError:
            log.warning(
                "No xtb-python installation found. install via `conda install xtb-python`"
            )
            raise

        calc = TBLite(
            method=self.method,
            charge=self.charge,
            multiplicity=self.multiplicity,
            accuracy=self.accuracy,
            electronic_temperature=self.electronic_temperature,
            max_iterations=self.max_iterations,
            initial_guess=self.initial_guess,
            mixer_damping=self.mixer_damping,
            spin_polarization=self.spin_polarization,
        )
        return calc
