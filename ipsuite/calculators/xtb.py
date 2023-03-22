import logging

import tqdm
import zntrack

from ipsuite import base

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
        self.atoms = self.get_data()
        print(self.atoms)

        calculator = self.calc

        for atom in tqdm.tqdm(self.atoms):
            atom.calc = calculator
            atom.get_potential_energy()

    @property
    def calc(self):
        """Get an xtb ase calculator."""
        try:
            from xtb.ase.calculator import XTB
        except ImportError:
            log.warning(
                "No xtb-python installation found. install via `conda install xtb-python`"
            )

        xtb = XTB(method=self.method)
        return xtb
