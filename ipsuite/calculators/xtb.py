import logging

import zntrack 

log = logging.getLogger(__name__)


class xTBCalc(zntrack.Node):
    """Node for obtaining xTB calculators.

    Attributes
    ----------
    method: str
        xTB method to be used. Only "GFN1-xTB" supports PBC.
    """
    method: str =  zntrack.zn.params("GFN1-xTB")

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