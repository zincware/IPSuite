import logging

import tqdm
import zntrack

from ipsuite import base
from ipsuite.utils.ase_sim import freeze_copy_atoms

log = logging.getLogger(__name__)


class xTBSinglePoint(base.ProcessAtoms):
    """Node for labeling date with xTB and obtaining ASE calculators.

    Attributes
    ----------
    method: str
        xTB method to be used. Only "GFN1-xTB" supports PBC.
    """

    method: str = zntrack.zn.params("GFN1-xTB")

    def run(self):
        self.atoms = []

        calculator = self.get_calculator()

        for atom in tqdm.tqdm(self.get_data()):
            atom.calc = calculator
            atom.get_potential_energy()
            self.atoms.append(freeze_copy_atoms(atom))

    def get_calculator(self, **kwargs):
        """Get an xtb ase calculator."""
        try:
            from xtb.ase.calculator import XTB
        except ImportError:
            log.warning(
                "No xtb-python installation found. install via `conda install xtb-python`"
            )
            raise

        return XTB(method=self.method)
